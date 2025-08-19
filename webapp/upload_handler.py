
import os
import csv
import faiss
import cv2
import shutil
import numpy as np
from datetime import datetime
from werkzeug.utils import secure_filename
from core.similarity.hashing import ImageHasher
from core.similarity.faiss_interface import FAISSIndex

PHASH_SIMILARITY_THRESHOLD = 0.40
HIST_SIMILARITY_THRESHOLD = 0.20
COSINE_SIMILARITY_THRESHOLD = 0.40  # New threshold for cosine similarity

class FakeFile:
    """A simple class to mock the file object for file-like operations."""
    def __init__(self, filepath):
        self.filename = os.path.basename(filepath)
        self.filepath = filepath

    def save(self, destination):
        shutil.copy(self.filepath, destination)

def process_upload(files, upload_dir):
    """
    Process uploaded images, detect duplicates, and organize into clusters
    
    Args:
        files: List of file objects (with save() method and filename attribute)
        upload_dir: Directory to save uploaded files
    
    Returns:
        Dictionary with results: clusters, duplicates, and errors
    """
    hasher = ImageHasher()
    faiss_index = FAISSIndex()

    results = {'clusters': [], 'duplicates': [], 'errors': []}
    os.makedirs(upload_dir, exist_ok=True)
    
    # Database to store histograms and other metadata
    metadata_db = {}
    
    print(f"Processing {len(files)} files")
    
    # Step 1: Save all files to disk
    saved_paths = []
    for file in files:
        if not file or file.filename == '':
            continue

        filename = secure_filename(file.filename)
        save_path = os.path.join(upload_dir, filename)
        print(f"Saving {save_path}")

        if not os.path.exists(save_path):
            file.save(save_path)
        
        saved_paths.append(save_path)

    print(f"Saved {len(saved_paths)} files")

    # Step 2: Extract features from all images
    all_features = []
    all_paths = []
    
    for path in saved_paths:
        try:
            # Compute perceptual hash
            phash = hasher.compute_phash(path)
            print(f"pHash: {phash}")
            
            # Compute histogram for secondary verification
            hist = hasher.compute_histogram(path)
            print(f"Histogram sum: {np.sum(hist)}")
            
            # Convert phash to float32 array for FAISS
            phash_float = np.unpackbits(phash).astype(np.float32)
            
            # Ensure proper dimensionality for FAISS
            if len(phash_float) != faiss_index.dim:
                phash_float = np.resize(phash_float, faiss_index.dim)
            
            all_features.append(phash_float)
            all_paths.append(path)
            
            # Store histogram in metadata database
            metadata_db[path] = {
                'hist': hist,
                'phash': phash  # Store original hash for direct comparison
            }
            
            print(f"Processed {path}: hash shape = {phash_float.shape}")
            
        except Exception as e:
            print(f"[ERROR] Failed to process {path}: {str(e)}")
            results['errors'].append({'path': path, 'error': str(e)})
            continue

    if not all_features:
        return results  # No valid images found
    
    # Step 3: Build and query the index
    feature_array = np.array(all_features, dtype=np.float32)
    
    # Simple batch processing adaptation depending on number of images
    num_images = len(all_features)
    print(f"Building index for {num_images} images")
    
    if num_images <= 100:
        # For small datasets, use a simple flat index
        index = faiss.IndexFlatL2(faiss_index.dim)
        index.add(feature_array)
        k = min(5, num_images)  # Don't search for more neighbors than we have images
        distances, indices = index.search(feature_array, k)
    else:
        # For larger datasets, use IVFPQ
        nlist = min(int(np.sqrt(num_images)), 100)
        quantizer = faiss.IndexFlatL2(faiss_index.dim)
        index = faiss.IndexIVFPQ(quantizer, faiss_index.dim, nlist, 8, 8)
        index.train(feature_array)
        index.add(feature_array)
        index.nprobe = min(nlist // 2, 10)
        k = min(10, num_images)
        distances, indices = index.search(feature_array, k)
    
    # Step 4: Process search results to find duplicates
    clusters = {}  # Maps cluster_id -> list of images
    duplicated = set()  # Keep track of which images have been marked as duplicates
    
    for i, (path, dists, neighbors) in enumerate(zip(all_paths, distances, indices)):
        # Skip if already marked as a duplicate
        if path in duplicated:
            continue
            
        # Create a new cluster
        cluster_id = len(clusters)
        clusters[cluster_id] = [path]
        
        # Skip the first match (which is the image itself)
        for j, (dist, neighbor_idx) in enumerate(zip(dists[1:], neighbors[1:])):
            if neighbor_idx >= len(all_paths):
                continue
                
            neighbor_path = all_paths[neighbor_idx]
            
            # Skip if already marked as a duplicate
            if neighbor_path in duplicated:
                continue
                
            # Check similarity using perceptual hash distance
            # Small L2 distance indicates similar hashes
            if dist < (1.0 - PHASH_SIMILARITY_THRESHOLD) * faiss_index.dim:
                # Perform multiple similarity checks for better accuracy
                
                # 1. Compare histograms
                current_hist = metadata_db[path]['hist']
                neighbor_hist = metadata_db[neighbor_path]['hist']
                hist_similarity = hasher.compare_histograms(current_hist, neighbor_hist)
                
                # 2. Calculate cosine similarity between images
                try:
                    cosine_similarity = hasher.compute_cosine_similarity(path, neighbor_path)
                    print(f"Cosine similarity between {path} and {neighbor_path}: {cosine_similarity:.4f}")
                except Exception as e:
                    print(f"Error calculating cosine similarity: {e}")
                    cosine_similarity = 0.0
                
                # Calculate combined similarity score with weights
                combined_similarity = (
                    0.5 * (1.0 - dist/faiss_index.dim) +  # pHash similarity (0.5 weight)
                    0.3 * hist_similarity +                # Histogram similarity (0.3 weight)
                    0.2 * cosine_similarity                # Cosine similarity (0.2 weight)
                )
                
                print(f"Combined similarity score: {combined_similarity:.4f}")
                
                # If the required criteria are met (using stricter conditions with the new metric)
                if (hist_similarity > HIST_SIMILARITY_THRESHOLD and 
                    cosine_similarity > COSINE_SIMILARITY_THRESHOLD):
                    
                    # Mark as duplicate and add to results
                    duplicated.add(neighbor_path)
                    clusters[cluster_id].append(neighbor_path)
                    
                    # Log duplicate with the combined similarity score
                    log_duplicate(neighbor_path, {
                        'path': path, 
                        'score': combined_similarity,
                        'cosine_similarity': cosine_similarity
                    }, results)
    
    # Step 5: Create cluster directories and copy images
    for cluster_id, paths in clusters.items():
        # Only report non-duplicate images as clusters
        primary_image = paths[0]
        results['clusters'].append({
            'path': primary_image,
            'cluster_id': cluster_id,
            'images': len(paths)
        })
        
        # Create cluster directory
        cluster_dir = os.path.join('clusters', f'cluster_{cluster_id}')
        os.makedirs(cluster_dir, exist_ok=True)
        
        # Copy all images in this cluster
        for img_path in paths:
            shutil.copy(img_path, os.path.join(cluster_dir, os.path.basename(img_path)))

    print(f"Found {len(results['clusters'])} unique clusters")
    print(f"Found {len(results['duplicates'])} duplicate images")
    print(f"Encountered {len(results['errors'])} errors")
    
    return results


def log_duplicate(path, match, results):
    # Updated to include cosine similarity information
    results['duplicates'].append({
        'original': match['path'],
        'duplicate': path,
        'similarity': match['score'],
        'cosine_similarity': match.get('cosine_similarity', 0.0),
        'timestamp': datetime.now().isoformat()
    })

    with open('duplicates.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            match['path'], 
            path, 
            f"{match['score']:.4f}", 
            f"{match.get('cosine_similarity', 0.0):.4f}",
            datetime.now().isoformat()
        ])

if __name__ == '__main__':
    files = [
        FakeFile(r'C:\Users\haris\OneDrive\Desktop\SRK1.jpeg'),
        FakeFile(r'C:\Users\haris\OneDrive\Desktop\SRK2.jpeg'),
        FakeFile(r'C:\Users\haris\OneDrive\Desktop\SRK3.jpeg'),
        FakeFile(r'C:\Users\haris\OneDrive\Desktop\SAL1.jpeg'),
        FakeFile(r'C:\Users\haris\OneDrive\Desktop\SAL2.jpeg'),
        FakeFile(r'C:\Users\haris\OneDrive\Desktop\SAL3.jpeg')
    ]
    upload_dir = 'uploaded_images'

    results = process_upload(files, upload_dir)

    print(f"Clusters: {len(results['clusters'])}")
    print(f"Duplicates: {len(results['duplicates'])}")
    print(f"Errors: {len(results['errors'])}")
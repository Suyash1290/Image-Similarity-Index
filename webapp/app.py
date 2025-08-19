from flask import Flask, request, render_template, jsonify, url_for, redirect, flash, send_from_directory
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

app = Flask(__name__)
app.secret_key = "image_similarity_secret_key"

# Configuration
UPLOAD_FOLDER = 'uploaded_images'
CLUSTERS_FOLDER = 'clusters'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
PHASH_SIMILARITY_THRESHOLD = 0.40
HIST_SIMILARITY_THRESHOLD = 0.20
COSINE_SIMILARITY_THRESHOLD = 0.40

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['CLUSTERS_FOLDER'] = CLUSTERS_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CLUSTERS_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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
    
    app.logger.info(f"Processing {len(files)} files")
    
    # Step 1: Save all files to disk
    saved_paths = []
    for file in files:
        if not file or not allowed_file(file.filename):
            continue

        filename = secure_filename(file.filename)
        save_path = os.path.join(upload_dir, filename)
        app.logger.info(f"Saving {save_path}")

        if not os.path.exists(save_path):
            file.save(save_path)
        
        saved_paths.append(save_path)

    app.logger.info(f"Saved {len(saved_paths)} files")

    # Step 2: Extract features from all images
    all_features = []
    all_paths = []
    
    for path in saved_paths:
        try:
            # Compute perceptual hash
            phash = hasher.compute_phash(path)
            
            # Compute histogram for secondary verification
            hist = hasher.compute_histogram(path)
            
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
            
            app.logger.debug(f"Processed {path}: hash shape = {phash_float.shape}")
            
        except Exception as e:
            app.logger.error(f"Failed to process {path}: {str(e)}")
            results['errors'].append({'path': path, 'error': str(e)})
            continue

    if not all_features:
        return results  # No valid images found
    
    # Step 3: Build and query the index
    feature_array = np.array(all_features, dtype=np.float32)
    
    # Simple batch processing adaptation depending on number of images
    num_images = len(all_features)
    app.logger.info(f"Building index for {num_images} images")
    
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
                    app.logger.debug(f"Cosine similarity between {path} and {neighbor_path}: {cosine_similarity:.4f}")
                except Exception as e:
                    app.logger.error(f"Error calculating cosine similarity: {e}")
                    cosine_similarity = 0.0
                
                # Calculate combined similarity score with weights
                combined_similarity = (
                    0.5 * (1.0 - dist/faiss_index.dim) +  # pHash similarity (0.5 weight)
                    0.3 * hist_similarity +                # Histogram similarity (0.3 weight)
                    0.2 * cosine_similarity                # Cosine similarity (0.2 weight)
                )
                
                app.logger.debug(f"Combined similarity score: {combined_similarity:.4f}")
                
                # If the required criteria are met
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
        cluster_name = f'cluster_{cluster_id}'
        
        results['clusters'].append({
            'path': primary_image,
            'relative_path': os.path.join(app.config['CLUSTERS_FOLDER'], cluster_name, os.path.basename(primary_image)),
            'cluster_id': cluster_id,
            'cluster_name': cluster_name,
            'images': len(paths)
        })
        
        # Create cluster directory
        cluster_dir = os.path.join(app.config['CLUSTERS_FOLDER'], cluster_name)
        os.makedirs(cluster_dir, exist_ok=True)
        
        # Copy all images in this cluster
        for img_path in paths:
            shutil.copy(img_path, os.path.join(cluster_dir, os.path.basename(img_path)))

    app.logger.info(f"Found {len(results['clusters'])} unique clusters")
    app.logger.info(f"Found {len(results['duplicates'])} duplicate images")
    app.logger.info(f"Encountered {len(results['errors'])} errors")
    
    return results

def log_duplicate(path, match, results):
    # Add to results object
    results['duplicates'].append({
        'original': match['path'],
        'duplicate': path,
        'similarity': match['score'],
        'cosine_similarity': match.get('cosine_similarity', 0.0),
        'timestamp': datetime.now().isoformat()
    })

    # Log to CSV file
    with open('duplicates.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        if os.stat('duplicates.csv').st_size == 0:  # Add header if file is empty
            writer.writerow(['Original', 'Duplicate', 'Similarity Score', 'Cosine Similarity', 'Timestamp'])
        writer.writerow([
            match['path'], 
            path, 
            f"{match['score']:.4f}", 
            f"{match.get('cosine_similarity', 0.0):.4f}",
            datetime.now().isoformat()
        ])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'files[]' not in request.files:
        flash('No file part')
        return redirect(request.url)
        
    files = request.files.getlist('files[]')
    
    if not files or files[0].filename == '':
        flash('No selected file')
        return redirect(request.url)
        
    valid_files = [f for f in files if f and allowed_file(f.filename)]
    
    if not valid_files:
        flash('No valid image files selected')
        return redirect(request.url)
        
    results = process_upload(valid_files, app.config['UPLOAD_FOLDER'])
    
    # Return JSON response for AJAX or redirect to results page
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return jsonify(results)
    else:
        # Set results in session for display on results page
        # In a production app, you'd use a database or more robust storage
        return redirect(url_for('results'))

@app.route('/results')
def results():
    # Retrieve all clusters
    clusters = []
    if os.path.exists(app.config['CLUSTERS_FOLDER']):
        for cluster_dir in os.listdir(app.config['CLUSTERS_FOLDER']):
            if os.path.isdir(os.path.join(app.config['CLUSTERS_FOLDER'], cluster_dir)):
                cluster_path = os.path.join(app.config['CLUSTERS_FOLDER'], cluster_dir)
                images = [f for f in os.listdir(cluster_path) if allowed_file(f)]
                if images:
                    clusters.append({
                        'cluster_name': cluster_dir,
                        'representative': images[0],
                        'count': len(images),
                        'images': images
                    })
    
    # Read duplicates from CSV
    duplicates = []
    if os.path.exists('duplicates.csv'):
        with open('duplicates.csv', 'r') as f:
            reader = csv.reader(f)
            next(reader, None)  # Skip header
            for row in reader:
                if len(row) >= 5:
                    duplicates.append({
                        'original': row[0],
                        'duplicate': row[1],
                        'similarity': row[2],
                        'cosine_similarity': row[3],
                        'timestamp': row[4]
                    })
    
    return render_template('results.html', clusters=clusters, duplicates=duplicates)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/clusters/<cluster>/<filename>')
def cluster_file(cluster, filename):
    return send_from_directory(os.path.join(app.config['CLUSTERS_FOLDER'], cluster), filename)

if __name__ == '__main__':
    # Create duplicates.csv if it doesn't exist
    if not os.path.exists('duplicates.csv'):
        with open('duplicates.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Original', 'Duplicate', 'Similarity Score', 'Cosine Similarity', 'Timestamp'])
    
    app.run(debug=True)
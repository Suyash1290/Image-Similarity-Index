import argparse
import csv
import os
import sqlalchemy
from datetime import datetime
from core.similarity import ImageHasher, ClusterEngine, FAISSIndex

def process_batch(input_dir, phash_thresh=0.85, hist_thresh=0.75):
    # Initialize components
    hasher = ImageHasher()
    faiss_index = FAISSIndex()
    engine = create_engine('sqlite:///similarity.db')
    
    # Process images
    for img_path in scan_images(input_dir):
        try:
            # Extract features
            phash = hasher.compute_phash(img_path)
            hist = hasher.compute_histogram(img_path)
            
            # Check for duplicates
            matches = faiss_index.search(phash.reshape(1,-1), k=5)
            is_duplicate = process_matches(matches, hist, phash_thresh, hist_thresh)
            
            if is_duplicate:
                log_duplicate(img_path, matches[0])
            else:
                save_to_db(img_path, phash, hist)
                faiss_index.add_vectors(phash.reshape(1,-1))
                
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")

def process_matches(matches, hist, phash_thresh, hist_thresh):
    for match in matches:
        # Retrieve stored histogram
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT histogram FROM images 
                WHERE id = :id
            """), {'id': match.id})
            stored_hist = np.frombuffer(result.fetchone()[0], dtype=np.uint32)
        
        # Calculate histogram similarity
        hist_sim = 1 - cv2.compareHist(
            hist.astype(np.float32),
            stored_hist.astype(np.float32),
            cv2.HISTCMP_BHATTACHARYYA
        )
        
        if match.similarity > phash_thresh and hist_sim > hist_thresh:
            return True
    return False

def log_duplicate(img_path, original_match):
    with open('duplicates.csv', 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'original_path', 
            'duplicate_path',
            'similarity_score',
            'timestamp',
            'resolution'
        ])
        
        writer.writerow({
            'original_path': original_match.path,
            'duplicate_path': img_path,
            'similarity_score': original_match.similarity,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'resolution': f"{original_match.width}x{original_match.height}"
        })

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process image batch')
    parser.add_argument('-i', '--input', required=True, 
                       help='Input directory with images')
    parser.add_argument('-p', '--phash', type=float, default=0.85,
                       help='Perceptual hash threshold')
    parser.add_argument('-b', '--hist', type=float, default=0.75,
                       help='Histogram similarity threshold')
    
    args = parser.parse_args()
    process_batch(args.input, args.phash, args.hist)

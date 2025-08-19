from core.similarity.clustering import ClusterEngine
from core.similarity.hashing import ImageHasher
import os
import numpy as np

def test_similarity():
    # Test images
    test_dir = "C:/Users/haris/OneDrive/Desktop/Image-Similarity-System/tests/"
    
    # Create test pairs
    test_pairs = [
        # Similar images (should have high similarity)
        ("base.jpg", "scaled.jpg", "Similar (scaled)"),
        
        # Different images (should have low similarity)
        ("base.jpg", "real_image.jpeg", "Different subject"),
        ("scaled.jpg", "real_image.jpeg", "Different subject"),
        
        # Test self-similarity (should be 1.0)
        ("base.jpg", "base.jpg", "Identical")
    ]
    
    hasher = ImageHasher()
    engine = ClusterEngine()

    print("Testing Image Similarity with Multiple Metrics:")
    print("-" * 100)
    print(f"{'Image 1':<15} {'Image 2':<15} {'Type':<20} {'pHash Sim':<10} {'Bits Diff':<10} {'Cosine Sim':<12} {'Hist Sim':<10}")
    print("-" * 100)
    
    for img1_name, img2_name, test_type in test_pairs:
        img1_path = os.path.join(test_dir, img1_name)
        img2_path = os.path.join(test_dir, img2_name)
        
        # Skip if any file doesn't exist
        if not os.path.exists(img1_path) or not os.path.exists(img2_path):
            print(f"Skipping {img1_name} vs {img2_name} - files not found")
            continue
        
        # Calculate hashes
        h1 = hasher.compute_phash(img1_path)
        h2 = hasher.compute_phash(img2_path)
        
        # Calculate raw bit differences
        bits1 = np.unpackbits(h1)
        bits2 = np.unpackbits(h2)
        bits_different = np.count_nonzero(bits1 != bits2)
        
        # Calculate pHash similarity
        phash_similarity = engine.calculate_similarity(h1, h2)
        
        # Calculate cosine similarity
        try:
            cosine_similarity = hasher.compute_cosine_similarity(img1_path, img2_path)
        except Exception as e:
            print(f"Error calculating cosine similarity: {e}")
            cosine_similarity = 0.0
            
        # Calculate histogram similarity
        hist1 = hasher.compute_histogram(img1_path)
        hist2 = hasher.compute_histogram(img2_path)
        hist_similarity = hasher.compare_histograms(hist1, hist2)
        
        # Calculate weighted combined similarity
        combined_similarity = (
            0.5 * phash_similarity +
            0.3 * hist_similarity +
            0.2 * cosine_similarity
        )
        
        print(f"{img1_name:<15} {img2_name:<15} {test_type:<20} {phash_similarity:.4f}    {bits_different:<10} {cosine_similarity:.4f}      {hist_similarity:.4f}")
    
    print("\nDetailed Similarity Analysis (Combined Metrics):")
    print("-" * 80)
    print(f"{'Image 1':<15} {'Image 2':<15} {'Type':<20} {'Combined Similarity':<20}")
    print("-" * 80)
    
    for img1_name, img2_name, test_type in test_pairs:
        img1_path = os.path.join(test_dir, img1_name)
        img2_path = os.path.join(test_dir, img2_name)
        
        # Skip if any file doesn't exist
        if not os.path.exists(img1_path) or not os.path.exists(img2_path):
            continue
            
        # Calculate all similarities
        h1 = hasher.compute_phash(img1_path)
        h2 = hasher.compute_phash(img2_path)
        phash_similarity = engine.calculate_similarity(h1, h2)
        
        try:
            cosine_similarity = hasher.compute_cosine_similarity(img1_path, img2_path)
        except Exception as e:
            cosine_similarity = 0.0
            
        hist1 = hasher.compute_histogram(img1_path)
        hist2 = hasher.compute_histogram(img2_path)
        hist_similarity = hasher.compare_histograms(hist1, hist2)
        
        # Calculate weighted combined similarity
        combined_similarity = (
            0.5 * phash_similarity +
            0.3 * hist_similarity +
            0.2 * cosine_similarity
        )
        
        print(f"{img1_name:<15} {img2_name:<15} {test_type:<20} {combined_similarity:.6f}")
    
    print("\nDetailed Hash Analysis:")
    print("-" * 60)
    for img_name in ["base.jpg", "scaled.jpg", "real_image.jpeg"]:
        img_path = os.path.join(test_dir, img_name)
        
        if not os.path.exists(img_path):
            continue
            
        h = hasher.compute_phash(img_path)
        bits = np.unpackbits(h)
        
        # Print hash as binary string
        binary = ''.join(['1' if b else '0' for b in bits])
        print(f"{img_name}: {binary}")
        
    print("\nPerformance Benchmark:")
    print("-" * 60)
    
    import time
    
    if os.path.exists(os.path.join(test_dir, "base.jpg")):
        img_path = os.path.join(test_dir, "base.jpg")
        
        # Benchmark pHash computation
        start_time = time.time()
        for _ in range(10):
            h = hasher.compute_phash(img_path)
        phash_time = (time.time() - start_time) / 10
        
        # Benchmark histogram computation
        start_time = time.time()
        for _ in range(10):
            hist = hasher.compute_histogram(img_path)
        hist_time = (time.time() - start_time) / 10
        
        # Benchmark cosine similarity computation
        start_time = time.time()
        for _ in range(10):
            sim = hasher.compute_cosine_similarity(img_path, img_path)
        cosine_time = (time.time() - start_time) / 10
        
        print(f"Average pHash computation time: {phash_time*1000:.2f} ms")
        print(f"Average histogram computation time: {hist_time*1000:.2f} ms")
        print(f"Average cosine similarity computation time: {cosine_time*1000:.2f} ms")

if __name__ == "__main__":
    test_similarity()
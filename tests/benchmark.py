import timeit
import memory_profiler
import numpy as np
import cv2
import ctypes
from core.similarity.hashing import ImageHasher

def benchmark_feature_extraction():
    hasher = ImageHasher()
    
    # Test with different image sizes
    test_images = [
        ('small', 'tests/test_images/512X512.jpeg'),  
        ('medium', 'tests/test_images/1024X1024.jpeg'),
        ('large', 'tests/test_images/4096X4096.jpeg')
    ]
    
    print("Benchmarking Feature Extraction:")
    print("{:<10} {:<15} {:<15} {:<10}".format(
        'Size', 'Time (ms)', 'Memory (MB)', 'NEON'))
    
    for name, path in test_images:
        # Preload the image to eliminate I/O overhead from timing
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        cv2.resize(img, (8, 8))  # Warm up the cv2 resize function
        
        # With NEON - using preloaded image
        def compute_with_neon():
            resized = cv2.resize(img, (8, 8), interpolation=cv2.INTER_AREA)
            phash = np.zeros(8, dtype=np.uint8)
            hasher.neon_phash.neon_phash(
                resized.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                phash.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                64
            )
            return phash
            
        # Without NEON
        def compute_without_neon():
            resized = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)
            dct = cv2.dct(np.float32(resized) / 255.0)
            low_freq = dct[:8, :8]
            median = np.median(low_freq)
            return np.packbits(low_freq > median)
        
        # Benchmark with more iterations for accuracy
        time_neon = timeit.timeit(compute_with_neon, number=1000) * 1.0  # Convert to ms
        mem_neon = memory_profiler.memory_usage((compute_with_neon,), max_usage=True)
        
        # Without NEON 
        time_base = timeit.timeit(compute_without_neon, number=1000) * 1.0  # Convert to ms
        mem_base = memory_profiler.memory_usage((compute_without_neon,), max_usage=True)
        
        print(f"{name:<10} {time_neon:<15.4f} {mem_neon:<15.2f} {'Enabled':<10}")
        print(f"{name:<10} {time_base:<15.4f} {mem_base:<15.2f} {'Disabled':<10}")


def batch_process_example():
    """Example of using ImageHasher for batch processing with high performance"""
    hasher = ImageHasher()
    
    # Example of processing multiple images in a directory
    import os
    from concurrent.futures import ThreadPoolExecutor
    
    image_dir = "tests/test_images/"
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    # Process images in parallel
    hashes = {}
    
    def process_image(path):
        try:
            hash_value = hasher.compute_phash(path)
            return path, hash_value
        except Exception as e:
            print(f"Error processing {path}: {e}")
            return path, None
    
    # Use thread pool to process images in parallel
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = executor.map(process_image, image_paths)
        
        for path, hash_value in results:
            if hash_value is not None:
                hashes[path] = hash_value
    
    print(f"Successfully processed {len(hashes)} images")
    
    # Example of finding similar images
    if len(hashes) >= 2:
        paths = list(hashes.keys())
        for i in range(len(paths)):
            for j in range(i+1, len(paths)):
                similarity = hasher.compare_hashes(hashes[paths[i]], hashes[paths[j]])
                if similarity > 0.9:  # Threshold for similar images
                    print(f"Similar images found: {paths[i]} and {paths[j]} ({similarity:.2f})")


if __name__ == '__main__':
    benchmark_feature_extraction()
    # Uncomment to run the batch processing example
    # batch_process_example()
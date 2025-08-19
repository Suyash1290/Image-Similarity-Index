import cv2
import numpy as np
import os
from core.similarity.hashing import ImageHasher
import matplotlib.pyplot as plt

def test_histogram():
    """Test the histogram computation using a real grayscale image"""
    hasher = ImageHasher()
    
    # Set the path to your real image file.
    # Adjust the path to where your image is stored.
    test_dir = os.path.dirname(os.path.abspath(__file__))
    test_img_path = os.path.join(test_dir, "real_image.jpeg")
    
    # Load the real image in grayscale mode.
    test_img = cv2.imread(test_img_path, cv2.IMREAD_GRAYSCALE)
    
    if test_img is None:
        raise ValueError(f"Could not load image from: {test_img_path}")

    # Optionally, display basic info about the image.
    print(f"Loaded image size: {test_img.shape}")
    
    # Compute histogram using the NEON implementation.
    neon_hist = hasher.compute_histogram(test_img_path)
    
    # Compute reference histogram using OpenCV directly.
    ref_hist = cv2.calcHist([test_img], [0], None, [256], [0, 256])
    ref_hist = ref_hist.flatten().astype(np.uint32)
    
    # Print basic statistics.
    print(f"NEON histogram sum: {np.sum(neon_hist)}")
    print(f"OpenCV histogram sum: {np.sum(ref_hist)}")
    
    similarity_correl = cv2.compareHist(
        neon_hist.astype(np.float32),
        ref_hist.astype(np.float32),
        cv2.HISTCMP_CORREL
    )
    
    similarity_chisqr = cv2.compareHist(
        neon_hist.astype(np.float32),
        ref_hist.astype(np.float32),
        cv2.HISTCMP_CHISQR
    ) / np.sum(ref_hist)
    
    similarity_bhatta = cv2.compareHist(
        neon_hist.astype(np.float32),
        ref_hist.astype(np.float32),
        cv2.HISTCMP_BHATTACHARYYA
    )
    
    print(f"Histogram Correlation: {similarity_correl:.4f} (higher is better, 1.0 is perfect)")
    print(f"Chi-Square Distance: {similarity_chisqr:.4f} (lower is better, 0.0 is perfect)")
    print(f"Bhattacharyya Distance: {similarity_bhatta:.4f} (lower is better, 0.0 is perfect)")
    
    # Visualize histograms for debugging.
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.title("NEON Histogram")
    plt.bar(range(256), neon_hist)
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Count")
    
    plt.subplot(1, 2, 2)
    plt.title("OpenCV Reference Histogram")
    plt.bar(range(256), ref_hist)
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Count")
    
    plt.tight_layout()
    plot_path = os.path.join(test_dir, "histogram_comparison.png")
    plt.savefig(plot_path)
    
    # Detailed logging for a few selected bins.
    print("\nDetailed comparison of selected bins:")
    sample_bins = [0, 64, 128, 192, 255]
    for bin_idx in sample_bins:
        print(f"Bin {bin_idx}: NEON={neon_hist[bin_idx]}, OpenCV={ref_hist[bin_idx]}, "
              f"Ratio={neon_hist[bin_idx] / max(1, ref_hist[bin_idx]):.2f}")
    
    # Optionally assert a minimum correlation threshold.
    assert similarity_correl > 0.95, f"Histogram correlation mismatch: {similarity_correl:.4f}"
    
    print(f"Histogram plot saved to: {plot_path}")
    print("Histogram test passed!")

if __name__ == "__main__":
    test_histogram()

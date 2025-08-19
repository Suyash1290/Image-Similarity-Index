import cv2
import numpy as np
import ctypes
from os.path import join, dirname
import threading

# Thread-local storage for hash arrays
thread_local = threading.local()

class ImageHasher:
    """
    Highly optimized image hasher using NEON acceleration
    with improved histogram computation and cosine similarity
    """
    # Load DLLs once at class level
    dll_path = join(dirname(__file__), "..", "neon_optimized", "phash_neon.dll")
    histo_path = join(dirname(__file__), "..", "neon_optimized", "histogram_neon.dll")
    
    try:
        neon_phash = ctypes.CDLL(dll_path)
        neon_histogram = ctypes.CDLL(histo_path)
        
        # Define argument and return types
        neon_phash.neon_phash.argtypes = [
            ctypes.POINTER(ctypes.c_uint8),  # image data pointer
            ctypes.POINTER(ctypes.c_uint8),  # output hash pointer
            ctypes.c_int                     # image size
        ]
        neon_phash.neon_phash.restype = None
        
        neon_histogram.neon_histogram.argtypes = [
            ctypes.POINTER(ctypes.c_uint8),  # data pointer
            ctypes.c_int,                    # width
            ctypes.c_int,                    # height
            ctypes.POINTER(ctypes.c_uint32)  # histogram pointer
        ]
        neon_histogram.neon_histogram.restype = None
        
        # Add the new cosine similarity function
        neon_histogram.neon_image_cosine_similarity = getattr(neon_histogram, "neon_image_cosine_similarity", None)
        if neon_histogram.neon_image_cosine_similarity:
            neon_histogram.neon_image_cosine_similarity.argtypes = [
                ctypes.POINTER(ctypes.c_uint8),  # image1 data pointer
                ctypes.POINTER(ctypes.c_uint8),  # image2 data pointer
                ctypes.c_int,                    # width
                ctypes.c_int                     # height
            ]
            neon_histogram.neon_image_cosine_similarity.restype = ctypes.c_float
        
        _has_neon = True
    except (OSError, AttributeError) as e:
        print(f"NEON acceleration not available: {e}")
        _has_neon = False
    
    def __init__(self, hash_size=64, dct_size=16):
        """
        Initialize hasher with customizable parameters
        hash_size: size of the resulting hash in bits
        dct_size: size of the DCT coefficient matrix to use (larger provides better discrimination)
        """
        self.hash_size = hash_size
        self.dct_size = dct_size  # Default to 16x16 DCT for better discrimination
        
    def get_thread_hash_array(self):
        """Get thread-local hash array to avoid reallocations"""
        if not hasattr(thread_local, 'hash_array'):
            thread_local.hash_array = np.zeros(8, dtype=np.uint8)
        return thread_local.hash_array
    
    def compute_phash(self, image_path):
        """
        Compute perceptual hash using NEON-optimized C++ implementation or fallback
        to enhanced Python implementation if NEON not available
        """
        # Read image
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE | cv2.IMREAD_IGNORE_ORIENTATION)
        if img is None or img.size == 0:
            raise ValueError(f"Invalid image: {image_path}")
        
        if self._has_neon:
            # Use NEON implementation for 8x8 DCT
            # Resize image to 8x8 using INTER_AREA for downsampling
            resized = cv2.resize(img, (8, 8), interpolation=cv2.INTER_AREA)
            
            # Ensure array is contiguous
            if not resized.flags['C_CONTIGUOUS']:
                resized = np.ascontiguousarray(resized)
            
            # Get thread-local hash array
            phash = self.get_thread_hash_array()
            
            # Call NEON-optimized phash function
            self.neon_phash.neon_phash(
                resized.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                phash.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                self.hash_size
            )
            
            # Return a copy of the hash
            return phash.copy()
        else:
            # Use fallback with more discrimination power
            return self._compute_phash_enhanced(img)
    
    def _compute_phash_enhanced(self, img):
        """
        Enhanced perceptual hash implementation with better discrimination
        using larger DCT and higher frequency components
        """
        # Resize to square image for DCT (larger size than standard pHash)
        img_size = self.dct_size * 4  # Using 4x the DCT size for input
        resized = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
        
        # Convert to float and compute DCT
        dct = cv2.dct(np.float32(resized) / 255.0)
        
        # Use a subset of the DCT coefficients (more than standard pHash)
        # Include both low and some mid frequency components
        dct_subset = dct[:self.dct_size, :self.dct_size]
        
        # Skip DC component (0,0) which represents the average brightness
        dct_subset[0, 0] = 0
        
        # Compute median of selected frequency components
        median = np.median(dct_subset)
        
        # Create hash from coefficients
        binary_hash = dct_subset > median
        
        # Pack the binary values into a byte array
        if binary_hash.size < 64:  # Ensure we have enough bits
            # Pad with zeros if necessary
            padded = np.zeros((8, 8), dtype=bool)
            padded[:binary_hash.shape[0], :binary_hash.shape[1]] = binary_hash
            binary_hash = padded
            
        return np.packbits(binary_hash.flatten()[:self.hash_size])
    
    def compute_histogram(self, image_path):
        """
        Compute histogram using NEON-optimized C++ implementation or fallback to OpenCV
        Returns a 256-bin histogram of grayscale image
        """
        # Read image in grayscale mode
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None or img.size == 0:
            raise ValueError(f"Invalid image: {image_path}")
        
        # Ensure array is contiguous
        if not img.flags['C_CONTIGUOUS']:
            img = np.ascontiguousarray(img)
            
        hist = np.zeros(256, dtype=np.uint32)
        
        # Call neon_histogram function if available
        if self._has_neon:
            try:
                self.neon_histogram.neon_histogram(
                    img.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                    img.shape[1], img.shape[0],
                    hist.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))
                )
            except Exception as e:
                print(f"NEON histogram failed: {e}, falling back to OpenCV")
                hist = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten().astype(np.uint32)
        else:
            # Fallback to OpenCV histogram
            hist = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten().astype(np.uint32)
        
        return hist
    
    def compute_cosine_similarity(self, image_path1, image_path2):
        """
        Compute cosine similarity between two images using NEON-optimized implementation
        or fallback to Python implementation if NEON not available
        
        Returns a similarity score between 0 and 1 (higher is more similar)
        """
        # Read images in grayscale mode
        img1 = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(image_path2, cv2.IMREAD_GRAYSCALE)
        
        if img1 is None or img1.size == 0:
            raise ValueError(f"Invalid image: {image_path1}")
        if img2 is None or img2.size == 0:
            raise ValueError(f"Invalid image: {image_path2}")
            
        # Resize images to the same dimensions if they differ
        if img1.shape != img2.shape:
            # Use the smaller dimensions to avoid artifacts
            height = min(img1.shape[0], img2.shape[0])
            width = min(img1.shape[1], img2.shape[1])
            img1 = cv2.resize(img1, (width, height), interpolation=cv2.INTER_AREA)
            img2 = cv2.resize(img2, (width, height), interpolation=cv2.INTER_AREA)
        
        # Ensure arrays are contiguous
        if not img1.flags['C_CONTIGUOUS']:
            img1 = np.ascontiguousarray(img1)
        if not img2.flags['C_CONTIGUOUS']:
            img2 = np.ascontiguousarray(img2)
            
        # Use NEON implementation if available
        if self._has_neon and hasattr(self.neon_histogram, 'neon_image_cosine_similarity'):
            try:
                similarity = self.neon_histogram.neon_image_cosine_similarity(
                    img1.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                    img2.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                    img1.shape[1], img1.shape[0]
                )
                return float(similarity)
            except Exception as e:
                print(f"NEON cosine similarity failed: {e}, falling back to Python implementation")
                return self._compute_cosine_similarity_py(img1, img2)
        else:
            # Fallback to Python implementation
            return self._compute_cosine_similarity_py(img1, img2)
    
    def _compute_cosine_similarity_py(self, img1, img2):
        """
        Python implementation of cosine similarity between image histograms
        """
        hist1 = cv2.calcHist([img1], [0], None, [256], [0, 256]).flatten()
        hist2 = cv2.calcHist([img2], [0], None, [256], [0, 256]).flatten()
        
        # Normalize histograms
        hist1 = hist1.astype(np.float32)
        hist2 = hist2.astype(np.float32)
        
        # Calculate cosine similarity
        dot_product = np.dot(hist1, hist2)
        norm1 = np.linalg.norm(hist1)
        norm2 = np.linalg.norm(hist2)
        
        # Avoid division by zero
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def compare_histograms(self, hist1, hist2, method=cv2.HISTCMP_CORREL):
        """
        Compare two histograms using OpenCV's compareHist function
        Returns similarity score between 0 and 1 (higher is better for correlation)
        """
        score = cv2.compareHist(
            hist1.astype(np.float32), 
            hist2.astype(np.float32), 
            method
        )
        
        # For methods where lower is better, convert to a 0-1 scale where higher is better
        if method == cv2.HISTCMP_CHISQR or method == cv2.HISTCMP_BHATTACHARYYA:
            return 1.0 / (1.0 + score)
        
        return score
    
    def compare_hashes(self, hash1, hash2):
        """
        Compare two hashes using fast bitwise operations
        Returns similarity score between 0 (different) and 1 (identical)
        """
        # Use the built-in popcount function for fast bit counting
        xor_result = np.bitwise_xor(hash1, hash2)
        hamming_distance = np.unpackbits(xor_result).sum()
        
        # Convert to similarity score (0-1 scale)
        max_bits = self.hash_size  # Maximum number of bits that could be different
        return 1.0 - (hamming_distance / max_bits)
    
    def compute_combined_hash(self, image_path):
        """
        Compute a combined hash using both pHash and edge information
        for better discrimination between visually different images
        """
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None or img.size == 0:
            raise ValueError(f"Invalid image: {image_path}")
        
        # Compute perceptual hash
        phash = self.compute_phash(image_path)
        
        # Compute edge hash
        edges = cv2.Canny(img, 100, 200)
        edge_resized = cv2.resize(edges, (8, 8), interpolation=cv2.INTER_AREA)
        edge_hash = np.packbits(edge_resized > 128)
        
        # Combine hashes
        return np.concatenate((phash, edge_hash))
        
    def compare_combined_hashes(self, hash1, hash2):
        """
        Compare combined hashes with weighted approach
        """
        # Split hashes into perceptual and edge components
        phash1, edge_hash1 = hash1[:8], hash1[8:]
        phash2, edge_hash2 = hash2[:8], hash2[8:]
        
        # Compare perceptual hashes
        p_xor = np.bitwise_xor(phash1, phash2)
        p_dist = np.unpackbits(p_xor).sum()
        p_sim = 1.0 - (p_dist / 64.0)
        
        # Compare edge hashes
        e_xor = np.bitwise_xor(edge_hash1, edge_hash2)
        e_dist = np.unpackbits(e_xor).sum()
        e_sim = 1.0 - (e_dist / 64.0)
        
        # Weight perceptual hash more than edge hash (adjustable)
        return 0.7 * p_sim + 0.3 * e_sim
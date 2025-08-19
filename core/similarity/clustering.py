import numpy as np

class ClusterEngine:
    """
    Improved engine for calculating image similarity and clustering
    """
    def __init__(self, threshold=0.85):
        self.threshold = threshold  # Similarity threshold for considering images as similar
    
    def calculate_similarity(self, hash1, hash2):
        """
        Calculate properly calibrated similarity between two image hashes
        """
        if hash1 is None or hash2 is None:
            return 0.0
            
        # Convert to numpy arrays if not already
        hash1 = np.array(hash1)
        hash2 = np.array(hash2)
        
        # Convert to bit arrays
        bits1 = np.unpackbits(hash1)
        bits2 = np.unpackbits(hash2)
        
        # Calculate Hamming distance
        hamming_distance = np.count_nonzero(bits1 != bits2)
        
        # Properly calibrated similarity calculation for perceptual hashes
        # Based on research that suggests most similar images have <= 15 bit differences
        normalized_similarity = self._calibrated_similarity(hamming_distance)
        
        return normalized_similarity
    
    def _calibrated_similarity(self, hamming_distance):
        """
        Convert Hamming distance to a properly calibrated similarity score
        
        For perceptual hashing, research shows:
        - 0-5 bits: Near duplicates/same image (0.90-1.00)
        - 6-10 bits: Very similar images (0.80-0.89)
        - 11-15 bits: Similar images (0.70-0.79)
        - 16-20 bits: Somewhat similar (0.60-0.69)
        - 21-25 bits: Slightly similar (0.50-0.59)
        - >25 bits: Different images (<0.50)
        """
        if hamming_distance <= 5:
            # Near duplicates - high similarity
            return 0.90 + (0.10 * (5 - hamming_distance) / 5)
        elif hamming_distance <= 10:
            # Very similar images
            return 0.80 + (0.09 * (10 - hamming_distance) / 5)
        elif hamming_distance <= 15:
            # Similar images
            return 0.70 + (0.09 * (15 - hamming_distance) / 5)
        elif hamming_distance <= 20:
            # Somewhat similar
            return 0.60 + (0.09 * (20 - hamming_distance) / 5)
        elif hamming_distance <= 25:
            # Slightly similar
            return 0.50 + (0.09 * (25 - hamming_distance) / 5)
        elif hamming_distance <= 32:
            # Not very similar
            return 0.20 + (0.29 * (32 - hamming_distance) / 7)
        else:
            # Different images - low similarity
            # Scale remaining distance from 0 to 0.19
            return max(0, 0.20 * (64 - hamming_distance) / 32)
    
    def cluster_images(self, hashes, filenames=None):
        """
        Cluster images based on hash similarity
        
        Parameters:
        - hashes: List of image hashes
        - filenames: Optional list of filenames corresponding to hashes
        
        Returns:
        - List of clusters, where each cluster is a list of indices or filenames
        """
        if not hashes:
            return []
        
        n = len(hashes)
        use_filenames = filenames is not None and len(filenames) == n
        
        # Initialize clusters and visited flags
        clusters = []
        visited = [False] * n
        
        for i in range(n):
            if visited[i]:
                continue
                
            # Start a new cluster
            cluster = [filenames[i] if use_filenames else i]
            visited[i] = True
            
            # Find all similar images
            for j in range(i+1, n):
                if not visited[j]:
                    similarity = self.calculate_similarity(hashes[i], hashes[j])
                    if similarity >= self.threshold:
                        cluster.append(filenames[j] if use_filenames else j)
                        visited[j] = True
            
            # Add cluster to results
            clusters.append(cluster)
        
        return clusters
    
    def find_similar(self, phash, threshold=0.9):
        """Find visually similar images"""
        query = np.array([phash], dtype=np.float32)
        distances, indices = self.index.search(query, 5)
        return [
            {'path': self.paths[i], 'score': float(1/(1+d))}
            for i, d in zip(indices[0], distances[0])
            if d < 1-threshold and i != -1
        ]
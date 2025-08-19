import faiss
import numpy as np

class FAISSIndex:
    """
    A simplified FAISS index that adapts to dataset size
    """
    def __init__(self, dim=64):
        """
        Initialize a FAISS index
        
        Args:
            dim: Dimension of feature vectors (default 64 for unpacked phash)
        """
        self.dim = dim
        self.index = None
        self.trained = False
    
    def build_index(self, vectors, force_flat=False):
        """
        Build an appropriate index based on dataset size
        
        Args:
            vectors: Feature vectors to index
            force_flat: If True, always use a flat index
            
        Returns:
            Built FAISS index
        """
        num_vectors = len(vectors)
        
        if num_vectors < 100 or force_flat:
            # For small datasets, use simple flat index
            index = faiss.IndexFlatL2(self.dim)
            index.add(vectors)
            self.trained = True
        else:
            # For larger datasets, use IVFPQ
            nlist = min(int(np.sqrt(num_vectors)), 100)
            quantizer = faiss.IndexFlatL2(self.dim)
            index = faiss.IndexIVFPQ(quantizer, self.dim, nlist, 8, 8)
            
            # Train the index
            index.train(vectors)
            self.trained = True
            
            # Add vectors
            index.add(vectors)
            
            # Adjust search parameters
            index.nprobe = min(nlist // 2, 10)
        
        self.index = index
        return index
    
    def search(self, query_vectors, k=5):
        """
        Search for similar vectors
        
        Args:
            query_vectors: Query vectors
            k: Number of results to return
            
        Returns:
            distances, indices tuple
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index first.")
            
        return self.index.search(
            np.array(query_vectors, dtype=np.float32), 
            k
        )
    
    def save(self, filepath="faiss_index.idx"):
        """Save index to disk"""
        if self.index is not None:
            faiss.write_index(self.index, filepath)
    
    def load(self, filepath="faiss_index.idx"):
        """Load index from disk"""
        self.index = faiss.read_index(filepath)
        self.trained = True
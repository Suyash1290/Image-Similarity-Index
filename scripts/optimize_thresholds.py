import numpy as np
import optuna
from sklearn.metrics import precision_recall_curve
from core.similarity import ImageHasher, ClusterEngine
from sqlalchemy import create_engine, text

def optimize_thresholds():
    # Connect to database
    engine = create_engine('sqlite:///similarity.db')
    
    # Load sample data
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT phash, histogram FROM images 
            LIMIT 1000
        """))
        data = result.fetchall()
    
    # Split into features
    phashes = [np.frombuffer(row[0], dtype=np.uint8) for row in data]
    histograms = [np.frombuffer(row[1], dtype=np.uint32) for row in data]

    def objective(trial):
        # Suggest thresholds
        phash_thresh = trial.suggest_float('phash_thresh', 0.7, 0.95)
        hist_thresh = trial.suggest_float('hist_thresh', 0.6, 0.9)
        
        # Calculate similarity matrix
        precision, recall = calculate_metrics(phashes, histograms,
                                            phash_thresh, hist_thresh)
        
        # Optimize for F1 score
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        return f1

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)

    print(f"Best thresholds: {study.best_params}")
    print(f"Best F1 score: {study.best_value}")

def calculate_metrics(phashes, histograms, phash_thresh, hist_thresh):
    tp = fp = fn = 0
    
    for i in range(len(phashes)):
        for j in range(i+1, len(phashes)):
            phash_sim = np.mean(phashes[i] == phashes[j])
            hist_sim = 1 - cv2.compareHist(
                histograms[i].astype(np.float32),
                histograms[j].astype(np.float32),
                cv2.HISTCMP_BHATTACHARYYA
            )
            
            actual = (phash_sim > phash_thresh) and (hist_sim > hist_thresh)
            expected = (i//10 == j//10)  # Assume 10 similar per group
            
            if actual and expected: tp += 1
            elif actual: fp += 1
            elif expected: fn += 1
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    return precision, recall

if __name__ == '__main__':
    optimize_thresholds()

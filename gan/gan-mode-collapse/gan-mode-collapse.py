import numpy as np

def detect_mode_collapse(generated_samples, threshold=0.1):
    diversity_score = float(np.mean(np.std(generated_samples, axis=0)))
    return {
        "diversity_score": round(diversity_score, 4),
        "is_collapsed": bool(diversity_score < threshold)
    }
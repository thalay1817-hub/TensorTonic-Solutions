import numpy as np

def crop_and_concat(encoder_features: np.ndarray, decoder_features: np.ndarray) -> np.ndarray:
    _, H_e, W_e, _ = encoder_features.shape
    _, H_d, W_d, _ = decoder_features.shape
    
    start_H = (H_e - H_d) // 2
    start_W = (W_e - W_d) // 2
    
    cropped = encoder_features[:, start_H:start_H+H_d, start_W:start_W+W_d, :]
    return np.concatenate([cropped, decoder_features], axis=3)
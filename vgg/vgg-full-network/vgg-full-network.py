def vgg16(x: np.ndarray, config: list, conv_weights: list, conv_biases: list,
          W1, b1, W2, b2, W3, b3) -> np.ndarray:
    features = vgg_features(x, config, conv_weights, conv_biases)
    return vgg_classifier(features, W1, b1, W2, b2, W3, b3)
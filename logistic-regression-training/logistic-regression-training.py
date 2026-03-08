import numpy as np

def sigmoid(x):
    x = np.asarray(x, dtype=float)
    return 1 / (1 + np.exp(-x))


def train_logistic_regression(X, y, lr=0.1, steps=500):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    
    N, d = X.shape
    
    # Initialize parameters
    w = np.zeros(d)
    b = 0.0
    
    for i in range(steps):
        # Linear combination
        z = X @ w + b
        
        # Predictions
        p = sigmoid(z)
        
        # Gradients
        grad_w = X.T @ (p - y) / N
        grad_b = np.mean(p - y)
        
        # Update parameters
        w = w - lr * grad_w
        b = b - lr * grad_b
    
    return w, b

# Prediction function
def predict(X, w, b):
    X = np.asarray(X, dtype=float)
    probs = sigmoid(X @ w + b)
    return (probs >= 0.5).astype(int)

# Accuracy function
def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

# Example
X = [[0], [1], [2], [3]]
y = [0, 0, 1, 1]

w, b = train_logistic_regression(X, y, lr=0.1, steps=500)

y_pred = predict(X, w, b)
acc = accuracy(y, y_pred)

print("Weights:", w)
print("Bias:", b)
print("Predictions:", y_pred)
print("Accuracy:", acc)

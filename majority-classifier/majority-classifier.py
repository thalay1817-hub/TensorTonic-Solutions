import numpy as np

def majority_classifier(y_train, X_test):
    # convert to numpy array
    y_train = np.array(y_train)

    # find unique classes and their counts
    classes, counts = np.unique(y_train, return_counts=True)

    # find majority class
    majority_class = classes[np.argmax(counts)]

    # predict majority class for all test samples
    predictions = np.full(len(X_test), majority_class)

    return predictions.tolist()


# Example 1
y_train = [0,1,1,1,0]
X_test = [10,20,30]
print(majority_classifier(y_train, X_test))

# Example 2
y_train = [2,2,2,1,0]
X_test = [5,6]
print(majority_classifier(y_train, X_test))
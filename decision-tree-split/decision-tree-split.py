def gini(labels):
    total = len(labels)
    if total == 0:
        return 0
    counts = {}
    for l in labels:
        counts[l] = counts.get(l, 0) + 1
    return 1 - sum((c/total)**2 for c in counts.values())


def decision_tree_split(X, y):
    n = len(X)
    d = len(X[0])

    parent_gini = gini(y)

    best_gain = -1
    best_feature = None
    best_threshold = None

    for f in range(d):
        # unique sorted values of this feature
        vals = sorted(set(X[i][f] for i in range(n)))

        # try midpoints between consecutive values
        for i in range(len(vals) - 1):
            threshold = (vals[i] + vals[i+1]) / 2

            left_y = []
            right_y = []

            for j in range(n):
                if X[j][f] <= threshold:
                    left_y.append(y[j])
                else:
                    right_y.append(y[j])

            # skip invalid splits
            if len(left_y) == 0 or len(right_y) == 0:
                continue

            # weighted gini
            gini_split = (len(left_y)/n)*gini(left_y) + (len(right_y)/n)*gini(right_y)

            gain = parent_gini - gini_split

            # tie-breaking: smaller feature index then smaller threshold
            if (gain > best_gain or
               (gain == best_gain and (f < best_feature or
               (f == best_feature and threshold < best_threshold)))):
                best_gain = gain
                best_feature = f
                best_threshold = threshold

    return [best_feature, best_threshold]


# Example 1
X = [[1,5],[2,5],[3,5],[4,5]]
y = [0,0,1,1]
print(decision_tree_split(X,y))

# Example 2
X = [[1,1],[2,1],[1,10],[2,10]]
y = [0,0,1,1]
print(decision_tree_split(X,y))
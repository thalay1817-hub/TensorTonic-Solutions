import math

def cosine_embedding_loss(x1, x2, label, margin=0.0):
    # dot product
    dot = sum(a*b for a, b in zip(x1, x2))

    # norms
    norm1 = math.sqrt(sum(a*a for a in x1))
    norm2 = math.sqrt(sum(b*b for b in x2))

    # avoid division by zero
    if norm1 == 0 or norm2 == 0:
        cos_sim = 0
    else:
        cos_sim = dot / (norm1 * norm2)

    # compute loss
    if label == 1:
        loss = 1 - cos_sim
    else:  # label == -1
        loss = max(0, cos_sim - margin)

    return loss


# Example 1
x1 = [1,0,0]
x2 = [1,0,0]
print(cosine_embedding_loss(x1, x2, 1, 0.0))

# Example 2
x1 = [1,0,0]
x2 = [0,1,0]
print(cosine_embedding_loss(x1, x2, 1, 0.0))
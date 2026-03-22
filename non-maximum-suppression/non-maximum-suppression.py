def iou(boxA, boxB):
    # Intersection coordinates
    x1 = max(boxA[0], boxB[0])
    y1 = max(boxA[1], boxB[1])
    x2 = min(boxA[2], boxB[2])
    y2 = min(boxA[3], boxB[3])
    
    # Compute intersection area
    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    intersection = inter_w * inter_h
    
    # Areas of boxes
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    
    # Union
    union = areaA + areaB - intersection
    
    return intersection / union if union > 0 else 0


def nms(boxes, scores, iou_threshold):
    if not boxes:
        return []
    
    # Sort indices by score (descending)
    indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    
    selected = []
    
    while indices:
        current = indices.pop(0)  # highest score
        selected.append(current)
        
        remaining = []
        for i in indices:
            if iou(boxes[current], boxes[i]) < iou_threshold:
                remaining.append(i)
        
        indices = remaining
    
    return selected
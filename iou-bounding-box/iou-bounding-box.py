def iou(box_a, box_b):
    # Unpack coordinates
    x1_a, y1_a, x2_a, y2_a = box_a
    x1_b, y1_b, x2_b, y2_b = box_b

    # Intersection rectangle
    x_left = max(x1_a, x1_b)
    y_top = max(y1_a, y1_b)
    x_right = min(x2_a, x2_b)
    y_bottom = min(y2_a, y2_b)

    # Compute intersection width & height (avoid negative)
    inter_w = max(0, x_right - x_left)
    inter_h = max(0, y_bottom - y_top)
    intersection = inter_w * inter_h

    # Areas of both boxes
    area_a = max(0, x2_a - x1_a) * max(0, y2_a - y1_a)
    area_b = max(0, x2_b - x1_b) * max(0, y2_b - y1_b)

    # Union
    union = area_a + area_b - intersection

    # Handle edge case
    if union == 0:
        return 0.0

    return intersection / union
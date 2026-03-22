import numpy as np

def generate_anchors(feature_size, image_size, scales, aspect_ratios):
    stride = image_size / feature_size
    anchors = []
    
    for i in range(feature_size):        # rows
        for j in range(feature_size):    # cols
            # Center of the cell
            cx = (j + 0.5) * stride
            cy = (i + 0.5) * stride
            
            for s in scales:
                for r in aspect_ratios:
                    # Width and height
                    w = s * np.sqrt(r)
                    h = s / np.sqrt(r)
                    
                    # Box coordinates
                    x1 = cx - w / 2
                    y1 = cy - h / 2
                    x2 = cx + w / 2
                    y2 = cy + h / 2
                    
                    anchors.append([float(x1), float(y1), float(x2), float(y2)])
    
    return anchors   # <-- returns list
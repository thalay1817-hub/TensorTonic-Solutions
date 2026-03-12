import numpy as np

def morphological_op(image, kernel, operation="erode"):
    image = np.asarray(image, dtype=int)
    kernel = np.asarray(kernel, dtype=int)

    h, w = image.shape
    kh, kw = kernel.shape

    # Padding size
    pad_h = kh // 2
    pad_w = kw // 2

    # Zero padding
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')

    output = np.zeros((h, w), dtype=int)

    for i in range(h):
        for j in range(w):

            if operation == "erode":
                val = 1
                for m in range(kh):
                    for n in range(kw):
                        if kernel[m][n] == 1 and padded[i+m][j+n] == 0:
                            val = 0
                            break
                    if val == 0:
                        break
                output[i][j] = val

            elif operation == "dilate":
                val = 0
                for m in range(kh):
                    for n in range(kw):
                        if kernel[m][n] == 1 and padded[i+m][j+n] == 1:
                            val = 1
                            break
                    if val == 1:
                        break
                output[i][j] = val

    return output.tolist()


# Example 1 (Dilation)
image = [[0,0,0],[0,1,0],[0,0,0]]
kernel = [[1,1,1],[1,1,1],[1,1,1]]
print(morphological_op(image, kernel, "dilate"))

# Example 2 (Erosion)
image = [[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1]]
kernel = [[1,1,1],[1,1,1],[1,1,1]]
print(morphological_op(image, kernel, "erode"))
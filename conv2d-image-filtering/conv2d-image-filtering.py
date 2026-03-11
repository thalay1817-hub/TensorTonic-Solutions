def conv2d(image, kernel, stride=1, padding=0):
    H = len(image)
    W = len(image[0])
    kh = len(kernel)
    kw = len(kernel[0])

    # Step 1: Create padded image
    padded_h = H + 2 * padding
    padded_w = W + 2 * padding
    padded = [[0 for _ in range(padded_w)] for _ in range(padded_h)]

    for i in range(H):
        for j in range(W):
            padded[i + padding][j + padding] = image[i][j]

    # Step 2: Compute output size
    H_out = ((H + 2 * padding - kh) // stride) + 1
    W_out = ((W + 2 * padding - kw) // stride) + 1

    output = [[0.0 for _ in range(W_out)] for _ in range(H_out)]

    # Step 3: Perform convolution
    for i in range(H_out):
        for j in range(W_out):
            val = 0.0
            for m in range(kh):
                for n in range(kw):
                    val += padded[i*stride + m][j*stride + n] * kernel[m][n]
            output[i][j] = val

    return output


# Example 1
image = [[1,2,3],
         [4,5,6],
         [7,8,9]]

kernel = [[1,0],
          [0,1]]

print(conv2d(image, kernel, stride=1, padding=0))


# Example 2
image = [[1,2],
         [3,4]]

kernel = [[1,1],
          [1,1]]

print(conv2d(image, kernel, stride=1, padding=1))
def color_to_grayscale(image):
    h = len(image)
    w = len(image[0])

    gray = []

    for i in range(h):
        row = []
        for j in range(w):
            R = image[i][j][0]
            G = image[i][j][1]
            B = image[i][j][2]

            # Luminance formula
            y = 0.299 * R + 0.587 * G + 0.114 * B
            row.append(y)

        gray.append(row)

    return gray


# Example 1
image = [[[255, 0, 0]]]
print(color_to_grayscale(image))

# Example 2
image = [[[255, 0, 0], [0, 255, 0]],
         [[0, 0, 255], [255, 255, 255]]]

print(color_to_grayscale(image))
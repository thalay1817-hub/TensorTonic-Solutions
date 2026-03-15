def image_histogram(image):
    # Initialize histogram with 256 bins
    hist = [0] * 256
    
    # Traverse each pixel
    for row in image:
        for pixel in row:
            hist[pixel] += 1
            
    return hist


# Example 1
image = [[0, 1], [1, 2]]
print(image_histogram(image))
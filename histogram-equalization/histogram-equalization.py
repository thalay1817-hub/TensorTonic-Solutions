def histogram_equalize(image):
    # Flatten image to compute histogram
    flat = [pixel for row in image for pixel in row]
    total_pixels = len(flat)

    # Step 1: Histogram (0–255)
    hist = [0] * 256
    for pixel in flat:
        hist[pixel] += 1

    # Step 2: CDF
    cdf = [0] * 256
    cdf[0] = hist[0]
    for i in range(1, 256):
        cdf[i] = cdf[i - 1] + hist[i]

    # Step 3: Find cdf_min (first non-zero)
    cdf_min = next((c for c in cdf if c > 0), 0)

    # Step 4: Handle edge case (all pixels same)
    if total_pixels == cdf_min:
        return [[0 for _ in row] for row in image]

    # Step 5: Mapping function
    def map_pixel(v):
        return round((cdf[v] - cdf_min) / (total_pixels - cdf_min) * 255)

    # Step 6: Apply mapping
    result = []
    for row in image:
        result.append([map_pixel(pixel) for pixel in row])

    return result
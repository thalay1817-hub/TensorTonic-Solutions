import numpy as np

def random_crop(image: np.ndarray, crop_size: int = 224,
                crop_y: int = None, crop_x: int = None) -> np.ndarray:
    H, W, _ = image.shape
    if crop_y is None:
        crop_y = np.random.randint(0, H - crop_size + 1)
    if crop_x is None:
        crop_x = np.random.randint(0, W - crop_size + 1)
    return image[crop_y:crop_y + crop_size, crop_x:crop_x + crop_size, :]

def random_horizontal_flip(image: np.ndarray, p: float = 0.5,
                            flip_rand: float = None) -> np.ndarray:
    if flip_rand is None:
        flip_rand = np.random.random()
    return image[:, ::-1, :] if flip_rand < p else image
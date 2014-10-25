import numpy as np

def normalise_vector(v):
    v_n = v / np.sqrt(np.sum(v ** 2, axis=-1))[..., None]
    v_n[np.isnan(v_n)] = 0.0
    return v_n


def row_norm(v):
    return np.sqrt(np.sum(v ** 2, axis=-1))


def normalise_image(image):
    """
    For normalising an image that represents a set of vectors.
    """
    vectors = image.as_vector(keep_channels=True)
    return image.from_vector(normalise_vector(vectors))


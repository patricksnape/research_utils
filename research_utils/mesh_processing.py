import numpy as np
from scipy.ndimage.filters import median_filter
import cv2


def preprocess_image(image):
    # Copy the depth part of the image
    depth_pixels = image.pixels[..., 2].copy()
    depth_pixels = rescale_to_opencv_image(depth_pixels)
    filtered_depth_pixels = median_filter(depth_pixels, 5)

    # Build mask for floodfilling, this lets me ignore all the pixels
    # from the background and around the ears
    mask = np.zeros((depth_pixels.shape[0] + 2, depth_pixels.shape[1] + 2),
                    dtype=np.uint8)
    # Flood fill from top left
    cv2.floodFill(filtered_depth_pixels, mask, (0, 0),
                  (255, 255, 255), flags=cv2.FLOODFILL_MASK_ONLY)
    # Flood fill from top right
    cv2.floodFill(filtered_depth_pixels, mask, (depth_pixels.shape[1] - 1, 0),
                  (255, 255, 255), flags=cv2.FLOODFILL_MASK_ONLY)
    # Truncate and negate the flood filled areas to find the facial region
    floodfill_mask = (~mask.astype(np.bool))[1:-1, 1:-1]

    # Build a mask of the areas inside the face that need inpainting
    inpaint_mask = ~image.mask.mask & floodfill_mask
    # Inpaint the image and filter to smooth
    inpainted_pixels = cv2.inpaint(depth_pixels,
                                   inpaint_mask.astype(np.uint8),
                                   5, cv2.INPAINT_NS)
    inpainted_pixels = median_filter(inpainted_pixels, 5)

    # Back to depth pixels
    image.pixels[..., 2] = rescale_to_depth_image(image, inpainted_pixels)
    # Reset the mask!
    image.mask.pixels[..., 0] = ~np.isnan(image.pixels[..., 2])


def rescale(val, in_min, in_max, out_min, out_max):
    return out_min + (val - in_min) * ((out_max - out_min) / (in_max - in_min))


def rescale_to_opencv_image(depth_image):
    nan_max = np.nanmax(depth_image)
    nan_min = np.nanmin(depth_image)
    depth_image[np.isnan(depth_image)] = nan_min - 2
    rescaled_depth = rescale(depth_image,
                             np.nanmin(depth_image), nan_max,
                             0.0, 1.0)
    return (rescaled_depth * 255).astype(np.uint8)


def rescale_to_depth_image(original_image, opencv_image):
    nan_max = np.nanmax(original_image.pixels[..., 2])
    nan_min = np.nanmin(original_image.pixels[..., 2])
    depth_pixels = opencv_image.astype(np.float) / 255.0
    depth_pixels = rescale(depth_pixels, 0.0, 1.0, nan_min, nan_max)
    depth_pixels[np.isclose(np.nanmin(depth_pixels), depth_pixels)] = np.nan
    return depth_pixels
import cv2
import numpy as np

def color_mask(image, lower_rgb, upper_rgb):
    """
    Create a mask in specified color range

    Args:
        image (Mat): rgb image
        lower_rgb (list): low rgb value
        upper_rgb (list): high rgb value

    Returns:
        Mat: binary mask
    """
    lower_bound = np.array(lower_rgb, dtype=np.uint8)
    upper_bound = np.array(upper_rgb, dtype=np.uint8)
    mask = cv2.inRange(image, lower_bound, upper_bound)
    return mask

def exclude_color_mask(image, lower_rgb, upper_rgb):
    """
    Create an inverse mask in specified color range
    Args:
        image (Mat): rgb image
        lower_rgb (list): low rgb value
        upper_rgb (list): high rgb value
    Returns:
        Mat: binary inverse mask
    """
    lower_bound = np.array(lower_rgb, dtype=np.uint8)
    upper_bound = np.array(upper_rgb, dtype=np.uint8)
    mask = cv2.inRange(image, lower_bound, upper_bound)
    mask_inv = cv2.bitwise_not(mask)
    return mask_inv

def dilate(img, kernel_size, iterations=1):
    """
    Perform dilation on image : 
    morphological operation that increases the object area and is used to accentuate features in an image.
    Convolve an image with a kernel (a matrix of odd size) and replacing the image pixel with the maximum value under the kernel.
    Args:
        img (Mat): binary image
        kernel_size (int): size of the kernel
        iterations (int): number of iterations
    Returns:
        Mat: dilated image
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    img_dilate = cv2.dilate(img, kernel, iterations=iterations)
    return img_dilate

def erode(img, kernel_size, iterations=1):
    """
    Perform erosion on image :
    morphological operation that decreases the object area and is used to remove small-scale details from a binary image.
    Convolve an image with a kernel (a matrix of odd size) and replacing the image pixel with the minimum value under the kernel.
    Args:
        img (Mat): binary image
        kernel_size (int): size of the kernel
        iterations (int): number of iterations
    Returns:
        Mat: eroded image
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    img_erode = cv2.erode(img, kernel, iterations=iterations)
    return img_erode

def opening(img, kernel_size):
    """
    Perform opening on image :
    morphological operation that removes small objects from the foreground (usually taken as the bright pixels) of an image,
    placing them in the background, while preserving the shape and size of larger objects in the image.
    It is achieved by performing erosion followed by dilation using the same structuring element for both operations.
    Args:
        img (Mat): binary image
        kernel_size (int): size of the kernel
    Returns:
        Mat: opened image
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    img_opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    return img_opening

def closing(img, kernel_size):
    """
    Perform closing on image :
    morphological operation that fills small holes and gaps in the foreground (usually taken as the bright pixels) of an image,
    while preserving the shape and size of larger objects in the image.
    It is achieved by performing dilation followed by erosion using the same structuring element for both operations.
    Args:
        img (Mat): binary image
        kernel_size (int): size of the kernel
    Returns:
        Mat: closed image
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    img_closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    return img_closing


def external_border(frame, lower_color, upper_color):
    """
    Find the external border of the chessboard in the image.
    Args:
        frame (Mat): input rgb image
        lower_color (list): lower bound of the color in rgb
        upper_color (list): upper bound of the color in rgb
    Returns:
        hull_shrink (ndarray): shrunk convex hull of the chessboard
        combined_contour (ndarray): combined contour of the chessboard
    """
    mask_white = color_mask(frame, lower_color, upper_color)

    mask_white = dilate(erode(mask_white, 2, 7), 10, 15)

    # external border of the chessboard
    edges = cv2.Canny(mask_white, 50, 150, apertureSize=3)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)

    # Combine all contours into one
    combined_contour = np.vstack(contours).squeeze()

    # Find the convex hull of the combined contour
    hull = cv2.convexHull(combined_contour)
    
    # Shrink the hull by 10%
    center = np.mean(hull, axis=0)
    hull_shrink = (hull - center) * 0.9 + center
    hull_shrink = hull_shrink.astype(np.int32)

    return hull_shrink, combined_contour
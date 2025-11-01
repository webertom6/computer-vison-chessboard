import cv2
import numpy as np

def color_mask(image, lower_rgb, upper_rgb):
    lower_bound = np.array(lower_rgb, dtype=np.uint8)
    upper_bound = np.array(upper_rgb, dtype=np.uint8)
    mask = cv2.inRange(image, lower_bound, upper_bound)
    return mask

def dilate(img, kernel_size, iterations=1):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    img_dilate = cv2.dilate(img, kernel, iterations=iterations)
    return img_dilate

def erode(img, kernel_size, iterations=1):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    img_erode = cv2.erode(img, kernel, iterations=iterations)
    return img_erode

def opening(img, kernel_size):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    img_opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    return img_opening

def closing(img, kernel_size):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    img_closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    return img_closing


def external_border(frame, lower_color, upper_color):
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
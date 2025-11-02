import cv2
from ultralytics import YOLO
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import sys
sys.path.append(os.path.abspath('.'))

from src.const import *
from src.morph import *

DISTANCE_HULL = 10

range_white = 45

lower_white = [max(0, c - range_white) for c in WHITE_CHESS]
upper_white = [min(240, c + range_white) for c in WHITE_CHESS]

"""
Custom YOLO train over Roboflow dataset on corner of chessboard
"""
# model = YOLO(r'C:\Users\Student11\Documents\git\ELEN-CV\project\yolo\yolo11n_custom6h.pt')
model = YOLO(r'.\weights\best.pt')
model2 = YOLO(r'.\weights\best11m.pt')

def double_detect_yolo(frame):

    frame_height, frame_width = frame.shape[:2]

    hull, combined_contour = external_border(frame, lower_white, upper_white)

    # draw the convex hull on the frame
    cv2.drawContours(frame, [hull], -1, (0, 255, 0), 3)

    # scale factor horizontal and vertical
    fx = 640 / frame_width
    fy = 640 / frame_height
    
    frame_resized = cv2.resize(frame, (640, 640), interpolation=cv2.INTER_AREA)

    # Result of the corners detection with YOLO
    results = model.track(source=frame, persist=True, verbose=False, show=False, tracker="bytetrack.yaml", conf=0.1, iou=0.1)
    results2 = model2.track(source=frame, persist=True, verbose=False, show=False, tracker="bytetrack.yaml", conf=0.1, iou=0.1)
    
    results.extend(results2)
    
    results_resized = model.track(source=frame_resized, persist=True, verbose=False, show=False, tracker="bytetrack.yaml", conf=0.1, iou=0.1)
    results2_resized = model2.track(source=frame_resized, persist=True, verbose=False, show=False, tracker="bytetrack.yaml", conf=0.1, iou=0.1)
    
    results_resized.extend(results2_resized)

    bbox_data = []
    
    for result in results_resized:
        for box in result.boxes:
            # Resized frame coordinates
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            
            # Map to original frame coordinates
            x1_original, y1_original = x1 / fx, y1 / fy
            x2_original, y2_original = x2 / fx, y2 / fy
            
            # Calculate the center of the bounding box
            cx_resized, cy_resized = (x1 + x2) / 2, (y1 + y2) / 2
            cx_original, cy_original = cx_resized / fx, cy_resized / fy
            
            distance = cv2.pointPolygonTest(hull, (cx_original, cy_original), True)
            if distance < -DISTANCE_HULL:
                continue  # Skip this prediction if the center is not inside the hull or within 10 pixels

            bbox_data.append((cx_original, cy_original, x1_original, y1_original, x2_original, y2_original))  # Store center and box dimensions

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2  # Center of the bounding box

            # Check if the center of the bounding box is inside the hull or within 10 pixels
            distance = cv2.pointPolygonTest(hull, (cx, cy), True)
            if distance < -DISTANCE_HULL:
                continue 
                
            bbox_data.append((cx, cy, x1, y1, x2, y2))

    return bbox_data, frame

def cluster_boxes(bbox):

    # Group bounding boxes using DBSCAN with a maximum distance of 100 pixels

    bbox_centers = np.array([[data[0], data[1]] for data in bbox])  # Only centers
    clustering = DBSCAN(eps=200, min_samples=1).fit(bbox_centers)  # eps = 100 pixels

    clusters = {}
    for idx, label in enumerate(clustering.labels_):
        if label not in clusters:
            clusters[label] = []
        # Add full bounding box data to the cluster
        clusters[label].append(bbox[idx])  

    # Compute the average bounding box for each cluster
    averaged_bboxes = []
    for cluster_id, boxes in clusters.items():
        # Average the coordinates of the bounding boxes in the cluster
        avg_x1 = int(np.mean([box[2] for box in boxes]))
        avg_y1 = int(np.mean([box[3] for box in boxes]))
        avg_x2 = int(np.mean([box[4] for box in boxes]))
        avg_y2 = int(np.mean([box[5] for box in boxes]))
        # Center of the averaged bounding box
        avg_cx, avg_cy = (avg_x1 + avg_x2) // 2, (avg_y1 + avg_y2) // 2  
        averaged_bboxes.append((avg_x1, avg_y1, avg_x2, avg_y2, avg_cx, avg_cy))

    return averaged_bboxes

def id_corner_chessboard(corners, clustered_bboxes, frame):

    for bbox in clustered_bboxes:

        x1, y1, x2, y2, cx, cy = bbox
        label_text = ""
        
        
        # Determine which corner the center belongs to
        if cx < frame.shape[1] / 2 and cy < frame.shape[0] / 2:
            corners["h8"][0], corners["h8"][1] = cx, cy
            label_text += " h8"
            
        elif cx >= frame.shape[1] / 2 and cy < frame.shape[0] / 2:
            corners["h1"][0], corners["h1"][1] = cx, cy
            label_text += " h1"

        elif cx < frame.shape[1] / 2 and cy >= frame.shape[0] / 2:
            corners["a8"][0], corners["a8"][1] = cx, cy
            label_text += " a8"

        elif cx >= frame.shape[1] / 2 and cy >= frame.shape[0] / 2:
            corners["a1"][0], corners["a1"][1] = cx, cy
            label_text += " a1"
            
    return corners

def corner_id_yolo(frame):

    # Initialize corner coordinates
    corners = {
        "h8": [None, None],
        "h1": [None, None],
        "a8": [None, None],
        "a1": [None, None]
    }

    bbox_data, frame = double_detect_yolo(frame)

    clustered_boxes = cluster_boxes(bbox_data)

    corners = id_corner_chessboard(corners, clustered_boxes, frame) 

    return corners

def warp_image(frame, corners, board_size=(800, 800)):
    
    # define the destination points for the transformation (clockwise from top-left)
    dest_points = np.array([
                            [0, 0],
                            [board_size[0] - 1, 0],
                            [board_size[0] - 1, board_size[1] - 1],
                            [0, board_size[1] - 1]
                        ], dtype="float32")

    # reorder corners for clockwise from top-left
    src_pts = np.array([
                        [corners["h8"][0], corners["h8"][1]],
                        [corners["h1"][0], corners["h1"][1]],
                        [corners["a1"][0], corners["a1"][1]],
                        [corners["a8"][0], corners["a8"][1]] 
                    ], dtype=np.float32)

    perspect_matrix = cv2.getPerspectiveTransform(src_pts, dest_points)
    warped_image = cv2.warpPerspective(frame, perspect_matrix, board_size)

    return warped_image

def segmented_grid_chessboard(warped_frame):

    h, w = warped_frame.shape[:2]

    # convert to grayscale and detect edges
    gray = cv2.cvtColor(warped_frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # automatic Canny thresholds based on median
    med = np.median(blur)
    lower = int(max(0, 0.66 * med))
    upper = int(min(255, 1.33 * med))
    edges = cv2.Canny(blur, lower, upper)

    # probabilistic Hough line transform
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=80,
                            minLineLength=60, maxLineGap=20)
    
    segs = lines.reshape(-1, 4)
    # compute direction vectors of each segment by (x2 - x1, y2 - y1)
    vecs = segs[:, 2:4] - segs[:, :2]
    print(vecs.shape)  # (N, 2) : N vectors, each with 2 components (dx, dy)

    # compute angle of each segment
    dx, dy = vecs[:, 0].astype(float), vecs[:, 1].astype(float)
    angles = np.arctan2(dy, dx)

    """
    extend detected nearly-vertical and nearly-horizontal segments 
    to full image borders to form a grid
    """
    
    class_tol = np.deg2rad(1)  # classify lines within 15 degrees as horizontal/vertical

    # get indices of lines with +- 15dg vert or horiz
    vert_idx = np.where(np.abs(np.abs(angles) - np.pi/2) < class_tol)[0]
    horiz_idx = np.where(np.abs(angles) < class_tol)[0]
    print(f"Number of vertical lines: {len(vert_idx)}")
    print(f"Number of horizontal lines: {len(horiz_idx)}")

    # use segs to define a x-y axis system based on midpoints of detected segments
    xs = np.array([(segs[i, 0] + segs[i, 2]) / 2.0 for i in vert_idx]) if vert_idx.size else np.array([])
    ys = np.array([(segs[i, 1] + segs[i, 3]) / 2.0 for i in horiz_idx]) if horiz_idx.size else np.array([])

    # cluster close lines to avoid duplicate nearby grid lines
    def cluster_1d(vals, eps=10):
        if vals.size == 0:
            return np.array([])
        vals = vals.reshape(-1, 1)
        clustering = DBSCAN(eps=eps, min_samples=1).fit(vals)
        clusters = []
        for lab in np.unique(clustering.labels_):
            members = vals[clustering.labels_ == lab].flatten()
            clusters.append(members.mean())
        return np.array(sorted(clusters))

    x_lines = cluster_1d(xs, eps=12)
    y_lines = cluster_1d(ys, eps=12)
    print(f"Number of vertical lines clustered: {len(x_lines)}")
    print(f"Number of horizontal lines clustered: {len(y_lines)}")

    if len(x_lines) < 9 or len(y_lines) < 9:
        # extrapolate missing lines to ensure at least 9 lines in each direction
        print("Not enough grid lines detected, extrapolating missing lines...")
        if len(x_lines) < 9 and len(x_lines) >= 2:
            x_step = np.median(np.diff(x_lines))
            while len(x_lines) < 9:
                if len(x_lines) < 9:
                    x_lines = np.insert(x_lines, 0, x_lines[0] - x_step)
                if len(x_lines) < 9:
                    x_lines = np.append(x_lines, x_lines[-1] + x_step)
        if len(y_lines) < 9 and len(y_lines) >= 2:
            y_step = np.median(np.diff(y_lines))
            while len(y_lines) < 9:
                if len(y_lines) < 9:
                    y_lines = np.insert(y_lines, 0, y_lines[0] - y_step)
                if len(y_lines) < 9:
                    y_lines = np.append(y_lines, y_lines[-1] + y_step)
    print(f"Final vertical lines: {len(x_lines)}")
    print(f"Final horizontal lines: {len(y_lines)}")

    # compute intersection points of the grid lines
    inters_pts = []
    for xv in x_lines:
        for yv in y_lines:
            xi = int(round(np.clip(xv, 0, w-1)))
            yi = int(round(np.clip(yv, 0, h-1)))
            inters_pts.append((xi, yi))

    external_pts = [
        (int(round(np.clip(x_lines[0], 0, w-1))), int(round(np.clip(y_lines[0], 0, h-1)))),
        (int(round(np.clip(x_lines[-1], 0, w-1))), int(round(np.clip(y_lines[0], 0, h-1)))),
        (int(round(np.clip(x_lines[0], 0, w-1))), int(round(np.clip(y_lines[-1], 0, h-1)))),
        (int(round(np.clip(x_lines[-1], 0, w-1))), int(round(np.clip(y_lines[-1], 0, h-1))))
    ]

    min_x, max_x, min_y, max_y = external_pts[0][0], external_pts[3][0], external_pts[0][1], external_pts[3][1]

    line_img = warped_frame.copy()

    # draw detected segments and grid lines from Hough transform
    for i, (x1, y1, x2, y2) in enumerate(segs):
        cv2.line(line_img, (int(x1), int(y1)), (int(x2), int(y2)), RED, 5)

    # draw clustered grid lines and intersection points
    for xv in x_lines:
        x = int(round(np.clip(xv, 0, w-1)))
        # cv2.line(line_img2, (x, 0), (x, h-1), BLUE, 2)
        cv2.line(warped_frame, (x, min_y), (x, max_y), BLUE, 2)

    for yv in y_lines:
        y = int(round(np.clip(yv, 0, h-1)))
        # cv2.line(warped_frame, (0, y), (w-1, y), BLUE, 2)
        cv2.line(warped_frame, (min_x, y), (max_x, y), BLUE, 2)

    for (xi, yi) in inters_pts:
        cv2.circle(warped_frame, (xi, yi), 6, ORANGE, -1)


    for i, (xi, yi) in enumerate(external_pts):
        cv2.circle(warped_frame, (xi, yi), 6, BLACK, 4)

    return inters_pts, external_pts, line_img, warped_frame

def cleaning_cell_chessboard(warped_frame):

    warped_frame = cv2.cvtColor(warped_frame, cv2.COLOR_BGR2RGB)

    range_white = 45        
    range_black = 45        
    lower_white = [max(0, c - range_white) for c in WHITE_CHESS]
    upper_white = [min(240, c + range_white) for c in WHITE_CHESS] 
    lower_black = [max(0, c - range_black) for c in BLACK_CHESS2] 
    upper_black = [min(240, c + range_black) for c in BLACK_CHESS2]

    # exclude (put at 0=black) white pixels in the specified range of chessboard
    exclude_white_mask = exclude_color_mask(warped_frame, lower_white, upper_white)

    # removes white areas from the image
    exclude_white_frame = cv2.bitwise_and(warped_frame, warped_frame, mask=exclude_white_mask)

    # exclude (put at 0=black) black pixels in the specified range of chessboard
    exclude_black_mask = exclude_color_mask(warped_frame, lower_black, upper_black)

    # removes black areas from the image
    exclude_black_frame = cv2.bitwise_and(warped_frame, warped_frame, mask=exclude_black_mask)

    exclude_white_black = cv2.bitwise_and(exclude_white_frame, exclude_black_frame)

    clean_exclude_white_black = dilate(erode(exclude_white_black, 3, 8), 3, 10)

    return clean_exclude_white_black

# !!!!! not use  mov, doesnt work, convert to avi
video_path = r'.\video\IMG_0485.avi'

cap = cv2.VideoCapture(video_path)

# Check if the video opened successfully
if not cap.isOpened():
    print("Error opening the video file")
    exit()
else:
    # Get frame rate information
    fps = int(cap.get(5))
    print("Frame Rate : ",fps,"frames per second")  
    
    # Get frame count, if you see -1, the video is not good
    frame_count = cap.get(7)
    print("Frame count : ", frame_count)

    frame_jump = 0.2 * frame_count
    print("Frame jump : ", frame_jump)
    
    # Get the width and height of frame
    frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print("Frame width : ", frame_width)
    print("Frame height : ", frame_height)

frame_idx = 0
jump = 0

while True:
    ret, frame = cap.read()

    if not ret:
        print("End of video stream or error reading the video.")
        break

    corners = corner_id_yolo(frame)

    if all([c[0] is not None and c[1] is not None for c in corners.values()]):
        warped_frame = warp_image(frame, corners)

        inters_pts, external_pts, line_img, warped_frame = segmented_grid_chessboard(warped_frame)

        cv2.imshow('Warped image segmented chessboard', warped_frame)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_idx += 1

    if frame_idx >= 3:

        frame_idx = 0
        # jump ahead by frame_jump frames
        current_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
        new_pos = min(frame_count - 1, current_pos + frame_jump)
        cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)

        # break
        
# Release the video capture
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()

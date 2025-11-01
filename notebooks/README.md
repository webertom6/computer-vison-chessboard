# notebooks

## main

### `1) yolo_corner_detect.ipynb`
- DETECTION OF CORNERS

### `2) warped_seg_chessboard.ipynb`
- PROJECTION USING CORNERS
- SEGMENTATION OF BOARD INTO 8x8 GRID

### `3) classification_pawns.ipynb`
- EXTRACT MEAN COLORS OF EACH CELL
- CLASSIFY EACH CELL TO DETECT PAWNS

## chessboard's corner detection

### `corner_detect.ipynb`
> find corner with Canny and Shi-Tomasi

-> not suitable, too much features 

### `corner_mask.ipynb`
> isolate the chessboard with color mask and morphological operation

-> useful for hull and morph operation

### `yolo_[...].ipynb`
> since ``cv2.findChessboardCornersSB()`` is useless is most of cases, a custom YOLO model trained on Robolflow to detect corners
- ``yolo_fix_guess.ipynb`` : improve prediction using a certain number of initial guesses (only valid fixed video)

- `yolo_hull_cluster.ipynb` : 
    - hull ensure the prediction is in a zone near the chessboard
    - predictions made using 2 YOLO model (resized and original)
    - similar predictions clustered in 4 categories/corners and averaged
    - identify corners (h8, h1, a1, a8) by quadrants of image

- `yolo_corner_detect.ipynb` : same but processing loop breaks in different step for clarity


## sticker detection
identify and locate blue and pink sticker on the edge of chessboard

### `stick_detect.ipynb`
> finding stickers using different colorspace (HSV)

### `mask_stick.ipynb`
> finding stickers using color mask (RGB) and morphological operation


## chessboard segmentation

### `seg_board.ipynb`
> segment using Probabilistic Hough Lines, identify vertical and horizontal lines, define an x-y axis, cluster close lines, get intersection points and define a 8x8 grid
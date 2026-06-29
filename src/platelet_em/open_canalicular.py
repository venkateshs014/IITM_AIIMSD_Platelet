"""Open Canalicular System (OCS) segmentation.

The Open Canalicular System manifests in EM images as vacuolar,
electron-lucent (bright) structures. This module provides an interactive
OpenCV tool, :func:`mark_oc_with_gui`, that isolates these bright regions
inside the previously segmented platelet body.

Workflow:

1. The platelet region of interest is recovered from the green plasma-membrane
   contour produced by :mod:`platelet_em.plasma_membrane` (via an HSV green
   mask that is filled to form the working region).
2. A high-intensity ("white") threshold (default ``155`` = ``35 + 120``) is
   applied within that region to capture structures significantly brighter than
   the cytoplasm.
3. Candidate contours are filtered by area to reject noise, and the surviving
   OCS contours are drawn in **red** on the output overlay.

The threshold and area limits are adjustable live via trackbars.
"""

import cv2
import numpy as np


def mark_oc_with_gui(image_path, output_folder, output_file_path):
    """Interactive detection of the Open Canalicular System (OCS).

    Uses an HSV green mask (to recover the plasma-membrane region) combined with
    a high-intensity white threshold to isolate electron-lucent OCS vacuoles.
    Threshold and area filtering are adjustable live.

    Args:
        image_path (str): Path to the annotated image whose green plasma-membrane
            contour defines the working region.
        output_folder (str): Directory associated with the run (kept for API
            compatibility with the batch driver).
        output_file_path (str): Full path where the OCS-annotated image is saved
            when the user presses ``s``.

    Controls:
        ``s`` save the current result, ``ESC`` exit.
    """

    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image at {image_path}")
        return

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Create window and trackbars
    window_name = 'Open Canalicular Detection'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    def nothing(x):
        pass

    cv2.createTrackbar('White Thresh', window_name, 35, 135, nothing)  # 0-135 maps to 120-255
    cv2.createTrackbar('Min Area', window_name, 7000, 100000, nothing)
    cv2.createTrackbar('Max Area', window_name, 1000000, 1000000, nothing)

    # Define green mask
    lower_green = np.array([40, 50, 50])
    upper_green = np.array([80, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    # Fill region inside green contour
    contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled_mask = np.zeros_like(green_mask)
    cv2.drawContours(filled_mask, contours, -1, 255, thickness=cv2.FILLED)

    masked_image = cv2.bitwise_and(image, image, mask=filled_mask)
    gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)

    while True:
        # Read trackbar values
        white_thresh_raw = cv2.getTrackbarPos('White Thresh', window_name)
        white_thresh = white_thresh_raw + 120  # Map 0-135 to 120-255
        min_area = cv2.getTrackbarPos('Min Area', window_name)
        max_area = cv2.getTrackbarPos('Max Area', window_name)

        # Threshold bright regions
        _, white_mask = cv2.threshold(gray, white_thresh, 255, cv2.THRESH_BINARY)

        # Find and filter contours
        white_contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_contours = [cnt for cnt in white_contours if min_area < cv2.contourArea(cnt) < max_area]

        # Draw filtered contours
        result = image.copy()
        cv2.drawContours(result, filtered_contours, -1, (0, 0, 255), 1)

        cv2.imshow(window_name, result)

        key = cv2.waitKey(10) & 0xFF
        if key == ord('s'):
            cv2.imwrite(output_file_path, result)
            print(f"Saved: {output_file_path}")
        elif key == 27:  # ESC to exit
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Interactive Open Canalicular System (OCS) detection."
    )
    parser.add_argument(
        "image_path",
        help="Path to the annotated image whose green plasma-membrane contour defines the region.",
    )
    parser.add_argument("output_file_path", help="Path to write the OCS-annotated result.")
    args = parser.parse_args()

    mark_oc_with_gui(args.image_path, output_folder=None, output_file_path=args.output_file_path)

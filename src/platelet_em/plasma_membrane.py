"""Plasma-membrane (ROI) and dense-granule segmentation.

This module implements the first stage of the automated pre-annotation
pipeline (Phase 1 of the methodology). It provides an interactive,
dual-threshold OpenCV tool, :func:`analyze_platelet`, that lets an operator:

1. Delineate the platelet plasma membrane by clicking a polygonal region of
   interest (ROI) around the cell.
2. Adjust a first intensity threshold (default ``135``) to capture the dark
   platelet body / plasma-membrane contour (drawn in **green**).
3. Adjust an optional second, stricter threshold (default ``120``) to isolate
   highly electron-dense structures such as dense granules (drawn in **blue**).

The tool supports zoom/pan navigation and live area filtering so that the
operator can reject debris and noise. The resulting binary mask and annotated
overlay are saved to disk and serve as ground-truth masks for the deep-learning
segmentation stage (Phase 2).

The thresholding logic captures darker platelet structures against the lighter
background plasma using an inverted binary threshold::

    M_plasma(x, y) = 255 if I_gray(x, y) < T_plasma else 0
"""

import os

import cv2
import numpy as np


def analyze_platelet(image_path, output_dir="."):
    """Analyze a platelet with interactive point selection and dual thresholds.

    Args:
        image_path (str): Path to the input EM image.
        output_dir (str): Directory where the binary mask and annotated overlay
            are written. Created if it does not exist.

    Returns:
        dict: Analysis results containing contour counts, contour areas, the
        selected ROI points, and the paths of the written output files.
    """

    # --- Load image ---
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")

    os.makedirs(output_dir, exist_ok=True)
    mask_path = os.path.join(output_dir, "debug_polygon_mask.png")
    final_path = os.path.join(output_dir, "platelet_dual_threshold_final.png")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    original_height, original_width = gray.shape

    # --- Global variables ---
    threshold_value_1 = 135  # First threshold
    threshold_value_2 = 120  # Second threshold
    zoom_factor = 1.0
    zoom_step = 0.1
    min_zoom, max_zoom = 0.2, 4.0
    pan_offset = [0, 0]
    drag_start = None
    clicked_points = []
    current_display = None
    current_binary_1 = None
    current_binary_2 = None
    phase = 1  # 1 = point selection, 2 = first threshold, 3 = second threshold
    mask = None
    window_width = 1200
    window_height = 800
    min_contour_area = 1000  # default minimum area
    max_contour_area = 800000  # default max area
    trackbars_initialized = False  # Flag to prevent early trackbar access

    # Store contours from both thresholds
    first_threshold_contours = []
    second_threshold_contours = []
    updating_display = False  # Flag to prevent callback conflicts

    def nothing(x):
        pass

    def update_threshold_1():
        """Update first binary image based on current threshold"""
        nonlocal current_binary_1
        _, current_binary_1 = cv2.threshold(gray, threshold_value_1, 255, cv2.THRESH_BINARY_INV)

    def update_threshold_2():
        """Update second binary image based on current threshold"""
        nonlocal current_binary_2
        _, current_binary_2 = cv2.threshold(gray, threshold_value_2, 255, cv2.THRESH_BINARY_INV)

    def get_display_image():
        """Get the properly zoomed and panned image"""
        # Calculate display dimensions based on zoom
        display_w = int(original_width * zoom_factor)
        display_h = int(original_height * zoom_factor)

        # Resize the image based on zoom factor
        zoomed_image = cv2.resize(image, (display_w, display_h), interpolation=cv2.INTER_LINEAR)

        # Apply pan offset - create a window view
        start_x = max(0, pan_offset[0])
        start_y = max(0, pan_offset[1])
        end_x = min(display_w, start_x + window_width)
        end_y = min(display_h, start_y + window_height)

        # Extract the visible portion
        if start_x < display_w and start_y < display_h:
            cropped = zoomed_image[start_y:end_y, start_x:end_x].copy()
        else:
            cropped = np.zeros((window_height, window_width, 3), dtype=np.uint8)

        return cropped, start_x, start_y

    def draw_contours_on_overlay(overlay, contours, color, view_start_x, view_start_y):
        """Helper function to draw contours on overlay with proper transformation"""
        for cnt in contours:
            # Transform contour points for current view
            transformed_cnt = []
            for point in cnt:
                x, y = point[0]
                # Scale to zoom level
                display_x = int(x * zoom_factor) - view_start_x
                display_y = int(y * zoom_factor) - view_start_y

                if 0 <= display_x < overlay.shape[1] and 0 <= display_y < overlay.shape[0]:
                    transformed_cnt.append([[display_x, display_y]])

            if transformed_cnt:
                transformed_cnt = np.array(transformed_cnt, dtype=np.int32)
                cv2.drawContours(overlay, [transformed_cnt], -1, color, 2)

    def update_display():
        """Update the display with current zoom, pan, and overlay"""
        nonlocal current_display

        # Get the display image with proper zoom and pan
        display_img, view_start_x, view_start_y = get_display_image()
        overlay = display_img.copy()

        # In phase 2, show first threshold contours only within the selected mask
        if phase == 2 and current_binary_1 is not None and mask is not None:
            # Create masked binary for contour detection
            masked_binary = cv2.bitwise_and(current_binary_1, current_binary_1, mask=mask)
            contours, _ = cv2.findContours(masked_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Filter contours by area
            valid_contours = [cnt for cnt in contours if min_contour_area < cv2.contourArea(cnt) < max_contour_area]

            if not valid_contours:
                # Use the polygon itself as the contour
                polygon_contour = np.array(clicked_points, dtype=np.int32).reshape(-1, 1, 2)
                valid_contours = [polygon_contour]

            # Draw first threshold contours in green
            draw_contours_on_overlay(overlay, valid_contours, (0, 255, 0), view_start_x, view_start_y)

        # In phase 3, show both threshold contours
        elif phase == 3 and mask is not None:
            # Draw saved first threshold contours in green
            draw_contours_on_overlay(overlay, first_threshold_contours, (0, 255, 0), view_start_x, view_start_y)

            # Draw current second threshold contours in blue
            if current_binary_2 is not None:
                masked_binary = cv2.bitwise_and(current_binary_2, current_binary_2, mask=mask)
                contours, _ = cv2.findContours(masked_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # Filter contours by area
                valid_contours = [cnt for cnt in contours if min_contour_area < cv2.contourArea(cnt) < max_contour_area]

                if not valid_contours:
                    # Use the polygon itself as the contour
                    polygon_contour = np.array(clicked_points, dtype=np.int32).reshape(-1, 1, 2)
                    valid_contours = [polygon_contour]

                # Draw second threshold contours in blue
                draw_contours_on_overlay(overlay, valid_contours, (255, 0, 0), view_start_x, view_start_y)

        # Draw clicked points and lines
        for i, pt in enumerate(clicked_points):
            # Transform point to display coordinates
            display_x = int(pt[0] * zoom_factor) - view_start_x
            display_y = int(pt[1] * zoom_factor) - view_start_y

            if 0 <= display_x < overlay.shape[1] and 0 <= display_y < overlay.shape[0]:
                cv2.circle(overlay, (display_x, display_y), 3, (0, 255, 0), -1)
                # Add point number
                cv2.putText(overlay, str(i + 1), (display_x + 5, display_y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # Draw lines between consecutive points
        if len(clicked_points) > 1:
            for i in range(len(clicked_points) - 1):
                pt1_x = int(clicked_points[i][0] * zoom_factor) - view_start_x
                pt1_y = int(clicked_points[i][1] * zoom_factor) - view_start_y
                pt2_x = int(clicked_points[i + 1][0] * zoom_factor) - view_start_x
                pt2_y = int(clicked_points[i + 1][1] * zoom_factor) - view_start_y

                if (0 <= pt1_x < overlay.shape[1] and 0 <= pt1_y < overlay.shape[0] and
                        0 <= pt2_x < overlay.shape[1] and 0 <= pt2_y < overlay.shape[0]):
                    cv2.line(overlay, (pt1_x, pt1_y), (pt2_x, pt2_y), (0, 255, 0), 2)

            # Close polygon if we have enough points
            if len(clicked_points) >= 3:
                pt1_x = int(clicked_points[-1][0] * zoom_factor) - view_start_x
                pt1_y = int(clicked_points[-1][1] * zoom_factor) - view_start_y
                pt2_x = int(clicked_points[0][0] * zoom_factor) - view_start_x
                pt2_y = int(clicked_points[0][1] * zoom_factor) - view_start_y

                if (0 <= pt1_x < overlay.shape[1] and 0 <= pt1_y < overlay.shape[0] and
                        0 <= pt2_x < overlay.shape[1] and 0 <= pt2_y < overlay.shape[0]):
                    cv2.line(overlay, (pt1_x, pt1_y), (pt2_x, pt2_y), (0, 255, 0), 2)

        # Add status text based on phase
        if phase == 1:
            text = f"PHASE 1 - Select Points | Zoom: {zoom_factor:.1f}x | Points: {len(clicked_points)}/3 min"
            color = (255, 255, 0)  # Yellow
        elif phase == 2:
            text = f"PHASE 2 - First Threshold: {threshold_value_1} (GREEN) | Area: {min_contour_area}-{max_contour_area} | Zoom: {zoom_factor:.1f}x"
            color = (0, 255, 0)  # Green
        else:  # phase == 3
            text = f"PHASE 3 - Second Threshold: {threshold_value_2} (BLUE) | GREEN: {threshold_value_1} | Zoom: {zoom_factor:.1f}x"
            color = (255, 0, 0)  # Blue (BGR format)

        cv2.putText(overlay, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Add instructions
        if phase == 1:
            instructions = "Left-click: Add point | Right-drag: Pan | +/-: Zoom | u: Undo | t: Next phase"
        elif phase == 2:
            instructions = "Trackbar: Adjust threshold | n: Save & next threshold | s: Skip to final"
        else:  # phase == 3
            instructions = "Trackbar: Adjust second threshold | s: Save final result"
        cv2.putText(overlay, instructions, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        current_display = overlay
        return overlay

    def click_event(event, x, y, flags, param):
        """Handle mouse events for clicking points and panning"""
        nonlocal current_display, drag_start, pan_offset

        if phase == 1 and event == cv2.EVENT_LBUTTONDOWN:
            # Convert display coordinates back to original image coordinates
            # Account for pan and zoom
            view_start_x = max(0, pan_offset[0])
            view_start_y = max(0, pan_offset[1])

            orig_x = int((x + view_start_x) / zoom_factor)
            orig_y = int((y + view_start_y) / zoom_factor)

            # Ensure coordinates are within bounds
            orig_x = max(0, min(orig_x, original_width - 1))
            orig_y = max(0, min(orig_y, original_height - 1))

            clicked_points.append((orig_x, orig_y))
            print(f"Point {len(clicked_points)}: ({orig_x}, {orig_y})")

            current_display = update_display()
            cv2.imshow("Platelet Analyzer", current_display)

        elif event == cv2.EVENT_RBUTTONDOWN:
            drag_start = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE and drag_start:
            dx = x - drag_start[0]
            dy = y - drag_start[1]

            # Update pan offset
            max_pan_x = max(0, int(original_width * zoom_factor) - window_width)
            max_pan_y = max(0, int(original_height * zoom_factor) - window_height)

            pan_offset[0] = max(0, min(pan_offset[0] - dx, max_pan_x))
            pan_offset[1] = max(0, min(pan_offset[1] - dy, max_pan_y))

            drag_start = (x, y)
            current_display = update_display()
            cv2.imshow("Platelet Analyzer", current_display)

        elif event == cv2.EVENT_RBUTTONUP:
            drag_start = None

    def trackbar_callback(val):
        """Callback for trackbar changes - only active when trackbars are initialized"""
        nonlocal threshold_value_1, threshold_value_2, min_contour_area, max_contour_area, updating_display

        # Only proceed if trackbars are properly initialized and not already updating
        if not trackbars_initialized or updating_display:
            return

        updating_display = True
        try:
            if phase == 2:
                threshold_value_1 = max(10, min(cv2.getTrackbarPos("Threshold", "Platelet Analyzer") * 5, 250))
                update_threshold_1()
            elif phase == 3:
                threshold_value_2 = max(10, min(cv2.getTrackbarPos("Threshold", "Platelet Analyzer") * 5, 250))
                update_threshold_2()

            min_contour_area = max(100, cv2.getTrackbarPos("Min Area", "Platelet Analyzer") * 100)
            max_contour_area = max(min_contour_area + 100, cv2.getTrackbarPos("Max Area", "Platelet Analyzer") * 100)

            if phase >= 2:
                current_display = update_display()
                cv2.imshow("Platelet Analyzer", current_display)
        except cv2.error as e:
            print(f"Trackbar error (ignoring): {e}")
        finally:
            updating_display = False

    # --- Initialize ---
    print("=== Dual Threshold Platelet Analysis Tool ===")
    print("Phase 1: Select points around platelet boundary")
    print("Phase 2: Adjust first threshold (GREEN contours)")
    print("Phase 3: Adjust second threshold (BLUE contours)")
    print("\nControls:")
    print("- Left-click: Select points (Phase 1 only)")
    print("- Trackbar: Adjust threshold (Phase 2 & 3)")
    print("- Right-click + drag: Pan view")
    print("- '+'/'-': Zoom in/out")
    print("- 'u': Undo last point (Phase 1 only)")
    print("- 't': Move to first threshold phase")
    print("- 'n': Save current threshold and move to second threshold")
    print("- 's': Save and process selection")
    print("- ESC: Exit")

    # STEP 1: Create window first
    cv2.namedWindow("Platelet Analyzer", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Platelet Analyzer", window_width, window_height)

    # STEP 2: Set mouse callback
    cv2.setMouseCallback("Platelet Analyzer", click_event)

    # STEP 3: Initialize threshold and display BEFORE creating trackbars
    update_threshold_1()
    update_threshold_2()
    current_display = update_display()
    cv2.imshow("Platelet Analyzer", current_display)

    # STEP 4: Create trackbars with initial values (this will trigger callbacks)
    cv2.createTrackbar("Threshold", "Platelet Analyzer", 65, 125, trackbar_callback)  # 27*5 = 160
    cv2.createTrackbar("Min Area", "Platelet Analyzer", 60, 500, trackbar_callback)    # 60*100 = 1000
    cv2.createTrackbar("Max Area", "Platelet Analyzer", 80000, 800000, trackbar_callback)  # 80000*100 = 800000

    # STEP 5: Now mark trackbars as initialized
    trackbars_initialized = True

    # STEP 6: Update values from trackbars now that they're initialized
    try:
        threshold_value_1 = max(10, min(cv2.getTrackbarPos("Threshold", "Platelet Analyzer") * 2, 250))
        min_contour_area = max(100, cv2.getTrackbarPos("Min Area", "Platelet Analyzer") * 100)
        max_contour_area = max(min_contour_area + 100, cv2.getTrackbarPos("Max Area", "Platelet Analyzer") * 100)
    except cv2.error:
        pass  # Use defaults if reading fails

    # --- Main interaction loop ---
    print(f"\n=== PHASE {phase}: POINT SELECTION ===")
    print("Click points around the platelet boundary (minimum 3 points)")
    print("Use zoom/pan controls to navigate. Press 't' when done selecting.")

    while True:
        key = cv2.waitKey(30) & 0xFF

        if key == 27:  # ESC - Exit
            cv2.destroyAllWindows()
            return {"status": "cancelled", "message": "Analysis cancelled by user"}

        elif key == ord('t'):  # Move to first threshold phase
            if phase == 1:
                if len(clicked_points) >= 3:
                    # Create mask from selected points
                    mask = np.zeros((original_height, original_width), dtype=np.uint8)
                    polygon = np.array([clicked_points], dtype=np.int32)
                    cv2.fillPoly(mask, [polygon], 255)

                    phase = 2
                    print(f"\n=== PHASE {phase}: FIRST THRESHOLD ADJUSTMENT (GREEN) ===")
                    print("Adjust the first threshold using the trackbar")
                    print("Press 'n' to save this threshold and move to second threshold")
                    print("Press 's' to skip second threshold and save final result")

                    # Set trackbar to current threshold value
                    cv2.setTrackbarPos("Threshold", "Platelet Analyzer", threshold_value_1 // 5)

                    # Update display to show threshold effects within mask
                    update_threshold_1()
                    current_display = update_display()
                    cv2.imshow("Platelet Analyzer", current_display)
                else:
                    print("Need at least 3 points to create mask. Continue selecting points.")
            else:
                print("Already past phase 1.")

        elif key == ord('n'):  # Save current threshold and move to next
            if phase == 2:
                # Save first threshold contours
                masked_binary = cv2.bitwise_and(current_binary_1, current_binary_1, mask=mask)
                contours, _ = cv2.findContours(masked_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # Filter contours by area
                valid_contours = [cnt for cnt in contours if min_contour_area < cv2.contourArea(cnt) < max_contour_area]

                if not valid_contours:
                    # Use the polygon itself as the contour
                    polygon_contour = np.array(clicked_points, dtype=np.int32).reshape(-1, 1, 2)
                    valid_contours = [polygon_contour]

                first_threshold_contours = valid_contours.copy()
                print(f"Saved {len(first_threshold_contours)} contours from first threshold ({threshold_value_1})")

                # Move to phase 3
                phase = 3
                print(f"\n=== PHASE {phase}: SECOND THRESHOLD ADJUSTMENT (BLUE) ===")
                print("Adjust the second threshold using the trackbar")
                print("First threshold contours are shown in GREEN")
                print("Second threshold contours are shown in BLUE")
                print("Press 's' when satisfied with both results")

                # Set trackbar to second threshold value
                cv2.setTrackbarPos("Threshold", "Platelet Analyzer", threshold_value_2 // 5)

                # Update display
                update_threshold_2()
                current_display = update_display()
                cv2.imshow("Platelet Analyzer", current_display)
            else:
                print("Only available in phase 2.")

        elif key == ord('+') or key == ord('='):  # Zoom in
            zoom_factor = min(zoom_factor + zoom_step, max_zoom)
            current_display = update_display()
            cv2.imshow("Platelet Analyzer", current_display)

        elif key == ord('-') or key == ord('_'):  # Zoom out
            zoom_factor = max(zoom_factor - zoom_step, min_zoom)
            current_display = update_display()
            cv2.imshow("Platelet Analyzer", current_display)

        elif key == ord('u') and phase == 1:  # Undo last point (only in phase 1)
            if clicked_points:
                removed_point = clicked_points.pop()
                print(f"Removed point: {removed_point}")
                current_display = update_display()
                cv2.imshow("Platelet Analyzer", current_display)

        elif key == ord('s'):  # Save and process
            if phase >= 2 and len(clicked_points) >= 3:
                if phase == 2:
                    # Save first threshold contours if not already saved
                    masked_binary = cv2.bitwise_and(current_binary_1, current_binary_1, mask=mask)
                    contours, _ = cv2.findContours(masked_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    valid_contours = [cnt for cnt in contours if min_contour_area < cv2.contourArea(cnt) < max_contour_area]

                    if not valid_contours:
                        polygon_contour = np.array(clicked_points, dtype=np.int32).reshape(-1, 1, 2)
                        valid_contours = [polygon_contour]

                    first_threshold_contours = valid_contours.copy()
                    print(f"Using single threshold with {len(first_threshold_contours)} contours")

                elif phase == 3:
                    # Save second threshold contours
                    masked_binary = cv2.bitwise_and(current_binary_2, current_binary_2, mask=mask)
                    contours, _ = cv2.findContours(masked_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    valid_contours = [cnt for cnt in contours if min_contour_area < cv2.contourArea(cnt) < max_contour_area]

                    if not valid_contours:
                        polygon_contour = np.array(clicked_points, dtype=np.int32).reshape(-1, 1, 2)
                        valid_contours = [polygon_contour]

                    second_threshold_contours = valid_contours.copy()
                    print(f"Using dual thresholds: {len(first_threshold_contours)} (GREEN) + {len(second_threshold_contours)} (BLUE) contours")

                print("\n=== PROCESSING COMPLETE ===")
                break
            elif phase == 1:
                print("Complete point selection first, then press 't' to move to threshold adjustment")
            else:
                print("Need at least 3 points and to be in threshold phase.")

    cv2.destroyAllWindows()

    # --- Process and save results ---
    if len(clicked_points) >= 3 and mask is not None:
        print("Processing and saving results...")

        # Save debug images
        cv2.imwrite(mask_path, mask)
        print(f"Saved: {mask_path}")

        # Create final output image
        output = image.copy()

        # Draw first threshold contours in green
        total_contours = 0
        all_areas = []

        if first_threshold_contours:
            for cnt in first_threshold_contours:
                area = cv2.contourArea(cnt)
                cv2.drawContours(output, [cnt], -1, (0, 255, 0), 2)  # Green
                total_contours += 1
                all_areas.append(("First", area, threshold_value_1))
                print(f"First Threshold Contour {total_contours}: Area = {area:.0f} pixels (threshold: {threshold_value_1})")

        # Draw second threshold contours in blue (if exists)
        if second_threshold_contours:
            for cnt in second_threshold_contours:
                area = cv2.contourArea(cnt)
                cv2.drawContours(output, [cnt], -1, (255, 0, 0), 2)  # Blue
                total_contours += 1
                all_areas.append(("Second", area, threshold_value_2))
                print(f"Second Threshold Contour {total_contours}: Area = {area:.0f} pixels (threshold: {threshold_value_2})")

        # Save final result
        cv2.imwrite(final_path, output)
        print(f"Saved: {final_path}")

        # Display final result
        scale_percent = 30
        display_width = int(original_width * scale_percent / 100)
        display_height = int(original_height * scale_percent / 100)
        resized_output = cv2.resize(output, (display_width, display_height))

        cv2.imshow("Final Dual Threshold Result", resized_output)
        print("\n=== DUAL THRESHOLD ANALYSIS COMPLETE ===")
        print(f"Total contours: {total_contours}")
        print(f"First threshold: {threshold_value_1} (GREEN)")
        if second_threshold_contours:
            print(f"Second threshold: {threshold_value_2} (BLUE)")
        print("Press any key to exit...")
        cv2.waitKey(5000)
        cv2.destroyAllWindows()

        # Return results
        return {
            "input_image": image_path,
            "status": "completed",
            "total_contour_count": total_contours,
            "first_threshold": {
                "value": threshold_value_1,
                "contour_count": len(first_threshold_contours),
                "areas": [area for _, area, _ in all_areas if _ == "First"]
            },
            "second_threshold": {
                "value": threshold_value_2,
                "contour_count": len(second_threshold_contours),
                "areas": [area for _, area, _ in all_areas if _ == "Second"]
            } if second_threshold_contours else None,
            "all_contour_details": all_areas,
            "min_contour_area": min_contour_area,
            "max_contour_area": max_contour_area,
            "selected_points": clicked_points,
            "output_files": {
                "mask": mask_path,
                "final_result": final_path
            }
        }

    else:
        return {"status": "cancelled", "message": "Analysis cancelled - need completed selection and threshold adjustment."}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Interactive dual-threshold plasma-membrane and dense-granule segmentation."
    )
    parser.add_argument("image_path", help="Path to the input EM image.")
    parser.add_argument(
        "-o", "--output-dir", default=".",
        help="Directory to write the mask and annotated overlay (default: current directory).",
    )
    args = parser.parse_args()

    result = analyze_platelet(args.image_path, output_dir=args.output_dir)
    print(result)

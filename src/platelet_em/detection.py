"""Automated platelet detection and quantitative feature extraction.

This is the **primary, non-interactive** Phase-1 pipeline. Given an EM image and
a set of intensity thresholds, :func:`detect_and_process_platelet` performs the
full ultrastructural analysis in one pass:

1. **Plasma membrane (green).** Adaptive inverse-binary thresholding followed by
   area filtering and a composite *size + centrality* score to pick the platelet
   contour within the field of view.
2. **Dense granules (blue).** A stringent dark threshold inside the platelet ROI.
3. **Open Canalicular System / OCS (red).** A high-intensity ("white") threshold
   inside the ROI capturing electron-lucent vacuoles.
4. **Texture & morphology.** GLCM homogeneity (dense granules) and contrast
   (OCS), and the plasma-membrane **fractal dimension** (box-counting).
5. **Spatial statistics.** Pairwise-distance clustering metrics per organelle
   class (mean/median/IQR and a clustering index = std / mean).

Per-image outputs include labeled composite masks, a measurements CSV, and a
text summary. These artifacts form the ground-truth masks and the morphological
feature table that drive the downstream grading model.

The helper functions (``calculate_*``, ``save_measurements_to_csv``,
``visualize_results``) are also importable individually.
"""

import os
import csv

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from skimage.feature import graycomatrix, graycoprops


def detect_and_process_platelet(image_path,
                                plasma_threshold=135,
                                plasma_min_area=1000000,
                                plasma_max_area=800000,
                                blue_threshold=70,
                                blue_min_area=10000,
                                blue_max_area=800000,
                                white_threshold=155,
                                red_min_area=7000,
                                red_max_area=1000000,
                                output_dir=None):
    """
    Complete platelet detection and analysis pipeline with labeled contours, area measurements,
    and clustering analysis using pairwise distances.

    Args:
        image_path: Path to input image
        plasma_threshold: Threshold for plasma membrane detection
        plasma_min_area: Min area for plasma membrane
        plasma_max_area: Max area for plasma membrane
        blue_threshold: Threshold for blue contours (dense granules)
        blue_min_area: Min area for blue contours
        blue_max_area: Max area for blue contours
        white_threshold: Threshold for white regions (OC structures)
        red_min_area: Min area for red contours
        red_max_area: Max area for red contours
        output_dir: Directory to save outputs (optional)

    Returns:
        Dictionary containing all masks, images, measurements, and clustering metrics
    """

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    original_height, original_width = gray.shape

    print(f"=== Complete Platelet Analysis Pipeline ===")
    print(f"Image: {os.path.basename(image_path)}")
    print(f"Image size: {original_width}x{original_height}")

    # ============================================
    # STEP 1: DETECT PLASMA MEMBRANE (GREEN)
    # ============================================
    print("\n--- Step 1: Plasma Membrane Detection ---")
    print(f"Threshold: {plasma_threshold}, Area range: {plasma_min_area}-{plasma_max_area}")

    # Apply threshold for plasma membrane
    _, binary_plasma = cv2.threshold(gray, plasma_threshold, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(binary_plasma, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours by area
    area_filtered = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if plasma_min_area < area < plasma_max_area:
            area_filtered.append(cnt)

    if not area_filtered:
        print("No plasma membrane detected!")
        return None

    # Filter by position and select best
    candidates = []
    for cnt in area_filtered:
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)

        # Calculate center
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            center_x = int(M["m10"] / M["m00"])
            center_y = int(M["m01"] / M["m00"])
        else:
            center_x = x + w // 2
            center_y = y + h // 2

        # Check position constraints
        edge_margin = min(original_width, original_height) * 0.05

        if (x > edge_margin and
            y > edge_margin and
            x + w < original_width - edge_margin and
            y + h < original_height - edge_margin and
            y + h < original_height * 0.85):

            # Calculate scores
            img_center_x = original_width // 2
            img_center_y = original_height // 2
            distance_from_center = np.sqrt((center_x - img_center_x)**2 + (center_y - img_center_y)**2)

            max_possible_area = original_width * original_height * 0.5
            normalized_area = min(area / max_possible_area, 1.0)
            size_score = normalized_area * 50

            max_distance = np.sqrt(original_width**2 + original_height**2) / 2
            normalized_distance = distance_from_center / max_distance
            centrality_score = (1 - normalized_distance) * 50

            final_score = size_score + centrality_score

            candidates.append({
                'contour': cnt,
                'area': area,
                'score': final_score
            })

    if not candidates:
        # Fallback to largest
        plasma_contour = max(area_filtered, key=cv2.contourArea)
    else:
        # Select best candidate
        best = max(candidates, key=lambda x: x['score'])
        plasma_contour = best['contour']

    plasma_area = cv2.contourArea(plasma_contour)
    print(f"Plasma membrane detected: Area={plasma_area:.0f}")

    # Create plasma membrane mask
    plasma_mask = np.zeros_like(gray)
    cv2.fillPoly(plasma_mask, [plasma_contour], 255)

    # Create plasma membrane overlay image
    plasma_overlay = image.copy()
    cv2.drawContours(plasma_overlay, [plasma_contour], -1, (0, 255, 0), 3)
    # Create image with white background and black contour for plasma membrane only
    plasma_outline_only = np.ones_like(image) * 255  # white background
    cv2.drawContours(plasma_outline_only, [plasma_contour], -1, (0, 0, 0), 3)  # black contour

    # ============================================
    # STEP 2: DETECT BLUE CONTOURS (DENSE GRANULES)
    # ============================================
    print("\n--- Step 2: Dense Granules Detection (Blue) ---")
    print(f"Threshold: {blue_threshold}, Area range: {blue_min_area}-{blue_max_area}")

    # Apply threshold for blue contours
    _, binary_blue = cv2.threshold(gray, blue_threshold, 255, cv2.THRESH_BINARY_INV)

    # Apply plasma mask to limit to platelet region
    binary_blue_masked = cv2.bitwise_and(binary_blue, binary_blue, mask=plasma_mask)

    # Find contours
    blue_contours, _ = cv2.findContours(binary_blue_masked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter by area and sort by area (largest first)
    blue_valid = []
    blue_areas = []
    blue_centroids = []
    for cnt in blue_contours:
        area = cv2.contourArea(cnt)
        if blue_min_area < area < blue_max_area:
            blue_valid.append(cnt)
            blue_areas.append(area)
            # Calculate centroid
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                blue_centroids.append((cx, cy))

    # Sort by area (largest first)
    if blue_valid:
        sorted_indices = np.argsort(blue_areas)[::-1]
        blue_valid = [blue_valid[i] for i in sorted_indices]
        blue_areas = [blue_areas[i] for i in sorted_indices]
        blue_centroids = [blue_centroids[i] for i in sorted_indices]

    print(f"Dense granules found: {len(blue_valid)}")

    # Create blue mask
    blue_mask = np.zeros_like(gray)
    cv2.fillPoly(blue_mask, blue_valid, 255)

    # Create blue overlay image
    blue_overlay = plasma_overlay.copy()
    cv2.drawContours(blue_overlay, blue_valid, -1, (255, 0, 0), 2)

    # ============================================
    # STEP 3: DETECT RED CONTOURS (OC STRUCTURES)
    # ============================================
    print("\n--- Step 3: Open Canalicular Structures Detection (Red) ---")
    print(f"Threshold: {white_threshold}, Area range: {red_min_area}-{red_max_area}")

    # Apply plasma mask to original image
    masked_for_red = cv2.bitwise_and(image, image, mask=plasma_mask)

    # Convert to grayscale and threshold for white regions
    gray_for_red = cv2.cvtColor(masked_for_red, cv2.COLOR_BGR2GRAY)
    _, white_binary = cv2.threshold(gray_for_red, white_threshold, 255, cv2.THRESH_BINARY)

    # Find contours
    red_contours, _ = cv2.findContours(white_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter by area and sort by area (largest first)
    red_valid = []
    red_areas = []
    red_centroids = []
    for cnt in red_contours:
        area = cv2.contourArea(cnt)
        if red_min_area < area < red_max_area:
            red_valid.append(cnt)
            red_areas.append(area)
            # Calculate centroid
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                red_centroids.append((cx, cy))

    # Sort by area (largest first)
    if red_valid:
        sorted_indices = np.argsort(red_areas)[::-1]
        red_valid = [red_valid[i] for i in sorted_indices]
        red_areas = [red_areas[i] for i in sorted_indices]
        red_centroids = [red_centroids[i] for i in sorted_indices]

    print(f"OC structures found: {len(red_valid)}")

    # Create red mask
    red_mask = np.zeros_like(gray)
    cv2.fillPoly(red_mask, red_valid, 255)

    # Create complete overlay image with all contours
    final_overlay = blue_overlay.copy()
    cv2.drawContours(final_overlay, red_valid, -1, (0, 0, 255), 2)

    # ============================================
    # STEP 4: CREATE COMPOSITE COLOR MASK WITH LABELS
    # ============================================
    print("\n--- Step 4: Creating Labeled Composite Color Mask ---")

    # Create a 3-channel color mask
    composite_mask = np.zeros((original_height, original_width, 3), dtype=np.uint8)

    # Set platelet area to white
    composite_mask[plasma_mask > 0] = [255, 255, 255]

    # Overlay blue contours (blue color)
    composite_mask[blue_mask > 0] = [255, 0, 0]  # Blue in BGR

    # Overlay red contours (red color)
    composite_mask[red_mask > 0] = [0, 0, 255]  # Red in BGR

    # Create labeled version
    labeled_composite = composite_mask.copy()

    # Add labels for dense granules (blue)
    for i, (cnt, centroid) in enumerate(zip(blue_valid, blue_centroids), 1):
        cx, cy = centroid
        # Add label with background for visibility
        label = f"{i}D"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2

        # Get text size for background
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        # Draw white background rectangle
        cv2.rectangle(labeled_composite,
                     (cx - text_width//2 - 2, cy - text_height//2 - 2),
                     (cx + text_width//2 + 2, cy + text_height//2 + 2),
                     (255, 255, 255), -1)

        # Draw black text
        cv2.putText(labeled_composite, label,
                   (cx - text_width//2, cy + text_height//2),
                   font, font_scale, (0, 0, 0), thickness)

    # Add labels for OC structures (red)
    for i, (cnt, centroid) in enumerate(zip(red_valid, red_centroids), 1):
        cx, cy = centroid
        # Add label with background for visibility
        label = f"{i}O"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2

        # Get text size for background
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        # Draw white background rectangle
        cv2.rectangle(labeled_composite,
                     (cx - text_width//2 - 2, cy - text_height//2 - 2),
                     (cx + text_width//2 + 2, cy + text_height//2 + 2),
                     (255, 255, 255), -1)

        # Draw black text
        cv2.putText(labeled_composite, label,
                   (cx - text_width//2, cy + text_height//2),
                   font, font_scale, (0, 0, 0), thickness)

    # ============================================
    # STEP 4.5: TEXTURE ANALYSIS
    # ============================================
    print("\n--- Step 4.5: Texture Analysis ---")

    # Calculate GLCM homogeneity for dense granules
    dg_homogeneity = calculate_glcm_homogeneity(image, blue_mask, blue_valid, "Dense Granules")

    # Calculate GLCM contrast for OC structures
    ocs_contrast = calculate_glcm_contrast(image, red_mask, red_valid, "OC Structures")

    # Calculate fractal dimension for plasma membrane
    plasma_fractal_dim = calculate_fractal_dimension(plasma_contour, image.shape)

    # Create image with white background and black contour for plasma membrane only
    plasma_outline_only = np.ones_like(image) * 255  # white background
    cv2.drawContours(plasma_outline_only, [plasma_contour], -1, (0, 0, 0), 3)  # black contour

    # ============================================
    # STEP 5: CLUSTERING ANALYSIS - PAIRWISE DISTANCES
    # ============================================
    print("\n--- Step 5: Clustering Analysis ---")

    # Calculate pairwise distances for dense granules
    dg_clustering_stats = calculate_clustering_metrics(blue_centroids, "Dense Granules")

    # Calculate pairwise distances for OC structures
    ocs_clustering_stats = calculate_clustering_metrics(red_centroids, "OC Structures")

    # ============================================
    # PREPARE MEASUREMENTS DATA
    # ============================================
    measurements = {
        'image_name': os.path.basename(image_path),
        'platelet_total_area': plasma_area,
        'plasma_fractal_dimension': plasma_fractal_dim,
        'dense_granules_count': len(blue_valid),
        'dense_granules_areas': blue_areas,
        'dense_granules_centroids': blue_centroids,
        'dense_granules_clustering': dg_clustering_stats,
        'dense_granules_homogeneity': dg_homogeneity,
        'ocs_count': len(red_valid),
        'ocs_areas': red_areas,
        'ocs_centroids': red_centroids,
        'ocs_clustering': ocs_clustering_stats,
        'ocs_contrast': ocs_contrast,
    }

    # ============================================
    # SAVE OUTPUTS IF DIRECTORY PROVIDED
    # ============================================
    results = {
        'plasma_mask': plasma_mask,
        'plasma_contour': plasma_contour,
        'plasma_overlay': plasma_overlay,
        'plasma_area': plasma_area,
        'blue_mask': blue_mask,
        'blue_contours': blue_valid,
        'blue_areas': blue_areas,
        'blue_centroids': blue_centroids,
        'blue_overlay': blue_overlay,
        'red_mask': red_mask,
        'red_contours': red_valid,
        'red_areas': red_areas,
        'red_centroids': red_centroids,
        'final_overlay': final_overlay,
        'composite_mask': composite_mask,
        'labeled_composite': labeled_composite,
        'measurements': measurements
    }

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(image_path))[0]

        # Save all outputs

        cv2.imwrite(os.path.join(output_dir, f"{base_name}_00_plasma_outline_black.png"), plasma_outline_only)    # Save the outline image if output_dir is provided
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_01_plasma_mask.png"), plasma_mask)
        # cv2.imwrite(os.path.join(output_dir, f"{base_name}_02_plasma_overlay.png"), plasma_overlay)
        # cv2.imwrite(os.path.join(output_dir, f"{base_name}_03_blue_mask.png"), blue_mask)
        # cv2.imwrite(os.path.join(output_dir, f"{base_name}_04_blue_overlay.png"), blue_overlay)
        # cv2.imwrite(os.path.join(output_dir, f"{base_name}_05_red_mask.png"), red_mask)
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_06_final_overlay.png"), final_overlay)
        # cv2.imwrite(os.path.join(output_dir, f"{base_name}_07_composite_mask.png"), composite_mask)
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_08_labeled_composite.png"), labeled_composite)

        # # Create and save distance histograms
        # if len(blue_centroids) > 1:
        #     create_distance_histogram(dg_clustering_stats['pairwise_distances'],
        #                             "Dense Granules",
        #                             os.path.join(output_dir, f"{base_name}_09_dg_distance_histogram.png"))

        # if len(red_centroids) > 1:
        #     create_distance_histogram(ocs_clustering_stats['pairwise_distances'],
        #                             "OC Structures",
        #                             os.path.join(output_dir, f"{base_name}_10_ocs_distance_histogram.png"))

        print(f"\nAll outputs saved to: {output_dir}")

        # Save CSV with measurements
        csv_path = os.path.join(output_dir, f"{base_name}_measurements.csv")
        save_measurements_to_csv(measurements, csv_path)

        # Save summary
        summary_path = os.path.join(output_dir, f"{base_name}_summary.txt")
        with open(summary_path, 'w') as f:
            f.write("Platelet Analysis Summary\n")
            f.write("=" * 50 + "\n")
            f.write(f"Image: {os.path.basename(image_path)}\n")
            f.write(f"Platelet total area: {plasma_area:.0f} pixels\n")
            f.write(f"Dense granules count: {len(blue_valid)}\n")
            f.write(f"OC structures count: {len(red_valid)}\n")
            f.write("\nDense Granules Areas:\n")
            for i, area in enumerate(blue_areas, 1):
                f.write(f"  {i}D: {area:.0f} pixels\n")
            f.write("\nOC Structures Areas:\n")
            for i, area in enumerate(red_areas, 1):
                f.write(f"  {i}O: {area:.0f} pixels\n")

            # Add clustering information
            f.write("\n" + "=" * 50 + "\n")
            f.write("CLUSTERING ANALYSIS\n")
            f.write("=" * 50 + "\n")

            if dg_clustering_stats['count'] > 1:
                f.write("\nDense Granules Clustering:\n")
                f.write(f"  Mean distance: {dg_clustering_stats['mean']:.2f} pixels\n")
                f.write(f"  Median distance: {dg_clustering_stats['median']:.2f} pixels\n")
                f.write(f"  Std deviation: {dg_clustering_stats['std']:.2f} pixels\n")
                f.write(f"  Min distance: {dg_clustering_stats['min']:.2f} pixels\n")
                f.write(f"  Max distance: {dg_clustering_stats['max']:.2f} pixels\n")
                f.write(f"  Q1 (25%): {dg_clustering_stats['q1']:.2f} pixels\n")
                f.write(f"  Q3 (75%): {dg_clustering_stats['q3']:.2f} pixels\n")
                f.write(f"  Clustering Index: {dg_clustering_stats['clustering_index']:.3f}\n")

            if ocs_clustering_stats['count'] > 1:
                f.write("\nOC Structures Clustering:\n")
                f.write(f"  Mean distance: {ocs_clustering_stats['mean']:.2f} pixels\n")
                f.write(f"  Median distance: {ocs_clustering_stats['median']:.2f} pixels\n")
                f.write(f"  Std deviation: {ocs_clustering_stats['std']:.2f} pixels\n")
                f.write(f"  Min distance: {ocs_clustering_stats['min']:.2f} pixels\n")
                f.write(f"  Max distance: {ocs_clustering_stats['max']:.2f} pixels\n")
                f.write(f"  Q1 (25%): {ocs_clustering_stats['q1']:.2f} pixels\n")
                f.write(f"  Q3 (75%): {ocs_clustering_stats['q3']:.2f} pixels\n")
                f.write(f"  Clustering Index: {ocs_clustering_stats['clustering_index']:.3f}\n")

            f.write("\nParameters used:\n")
            f.write(f"  Plasma threshold: {plasma_threshold}\n")
            f.write(f"  Blue threshold: {blue_threshold}\n")
            f.write(f"  White threshold: {white_threshold}\n")

            # Add texture analysis information
            f.write("\n" + "=" * 50 + "\n")
            f.write("TEXTURE ANALYSIS\n")
            f.write("=" * 50 + "\n")

            if plasma_fractal_dim is not None:
                f.write(f"\nPlasma Membrane Boundary Complexity:\n")
                f.write(f"  Fractal Dimension: {plasma_fractal_dim:.4f}\n")
                complexity = "Simple" if plasma_fractal_dim < 1.2 else "Moderate" if plasma_fractal_dim < 1.5 else "Complex"
                f.write(f"  Complexity Level: {complexity}\n")

            if dg_homogeneity['mean'] is not None:
                f.write(f"\nDense Granules Texture:\n")
                f.write(f"  GLCM Homogeneity: {dg_homogeneity['mean']:.4f} ± {dg_homogeneity['std']:.4f}\n")
                texture_type = "Heterogeneous" if dg_homogeneity['mean'] < 0.5 else "Moderate" if dg_homogeneity['mean'] < 0.8 else "Homogeneous"
                f.write(f"  Texture Type: {texture_type}\n")

            if ocs_contrast['mean'] is not None:
                f.write(f"\nOC Structures Texture:\n")
                f.write(f"  GLCM Contrast: {ocs_contrast['mean']:.4f} ± {ocs_contrast['std']:.4f}\n")
                contrast_level = "Low" if ocs_contrast['mean'] < 50 else "Moderate" if ocs_contrast['mean'] < 150 else "High"
                f.write(f"  Contrast Level: {contrast_level}\n")

    return results


def calculate_clustering_metrics(centroids, object_type):
    """
    Calculate pairwise distance statistics for clustering analysis

    Args:
        centroids: List of (x, y) tuples representing object centroids
        object_type: String describing the object type (for logging)

    Returns:
        Dictionary containing clustering statistics
    """

    if len(centroids) < 2:
        print(f"  {object_type}: Not enough objects for clustering analysis (found: {len(centroids)})")
        return {
            'count': len(centroids),
            'pairwise_distances': [],
            'mean': None,
            'median': None,
            'std': None,
            'min': None,
            'max': None,
            'q1': None,
            'q3': None,
            'clustering_index': None
        }

    # Convert centroids to numpy array
    points = np.array(centroids)

    # Calculate pairwise distances
    distances = pdist(points, metric='euclidean')

    # Calculate statistics
    mean_dist = np.mean(distances)
    median_dist = np.median(distances)
    std_dist = np.std(distances)
    min_dist = np.min(distances)
    max_dist = np.max(distances)
    q1 = np.percentile(distances, 25)
    q3 = np.percentile(distances, 75)

    # Calculate clustering index (coefficient of variation)
    # Lower values indicate more uniform spacing, higher values indicate clustering
    clustering_index = std_dist / mean_dist if mean_dist > 0 else 0

    print(f"  {object_type} Clustering Analysis:")
    print(f"    Objects: {len(centroids)}")
    print(f"    Mean distance: {mean_dist:.2f} pixels")
    print(f"    Median distance: {median_dist:.2f} pixels")
    print(f"    Clustering Index: {clustering_index:.3f} (lower=uniform, higher=clustered)")

    return {
        'count': len(centroids),
        'pairwise_distances': distances.tolist(),
        'mean': mean_dist,
        'median': median_dist,
        'std': std_dist,
        'min': min_dist,
        'max': max_dist,
        'q1': q1,
        'q3': q3,
        'clustering_index': clustering_index
    }


def create_distance_histogram(distances, object_type, output_path):
    """
    Create and save a histogram of pairwise distances

    Args:
        distances: List of pairwise distances
        object_type: String describing the object type
        output_path: Path to save the histogram
    """
    if not distances:
        return

    plt.figure(figsize=(10, 6))

    # Create histogram
    n_bins = min(30, max(5, len(distances) // 10))
    plt.hist(distances, bins=n_bins, edgecolor='black', alpha=0.7, color='blue')

    # Add statistics lines
    mean_val = np.mean(distances)
    median_val = np.median(distances)

    plt.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.1f}')
    plt.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.1f}')

    plt.xlabel('Distance (pixels)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(f'{object_type} - Pairwise Distance Distribution', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Add text box with statistics
    stats_text = f'Count: {len(distances)}\nMin: {np.min(distances):.1f}\nMax: {np.max(distances):.1f}\nStd: {np.std(distances):.1f}'
    plt.text(0.98, 0.97, stats_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=100)
    plt.close()


def calculate_glcm_homogeneity(image, mask, contours, object_type="Dense Granules"):
    """
    Calculate GLCM homogeneity for each object (dense granules)
    Homogeneity measures how close the distribution of elements in GLCM is to diagonal
    Higher values indicate more homogeneous texture
    """
    homogeneity_values = []

    for i, contour in enumerate(contours):
        # Get bounding box for the contour
        x, y, w, h = cv2.boundingRect(contour)

        # Extract ROI from image and mask
        roi = image[y:y+h, x:x+w]
        roi_mask = mask[y:y+h, x:x+w]

        # Create masked ROI (keep only pixels inside contour)
        masked_roi = cv2.bitwise_and(roi, roi, mask=roi_mask)

        if masked_roi.size == 0 or np.sum(roi_mask) == 0:
            continue

        # Convert to grayscale if needed
        if len(masked_roi.shape) == 3:
            masked_roi = cv2.cvtColor(masked_roi, cv2.COLOR_BGR2GRAY)

        # Get only the pixels inside the mask for normalization
        valid_pixels = masked_roi[roi_mask > 0]

        if len(valid_pixels) < 4:  # Need minimum pixels for GLCM
            continue

        # Normalize to 0-255 range for better GLCM computation
        min_val = np.min(valid_pixels)
        max_val = np.max(valid_pixels)
        if max_val > min_val:
            masked_roi = ((masked_roi - min_val) / (max_val - min_val) * 255).astype(np.uint8)

        # Apply mask again after normalization
        masked_roi = cv2.bitwise_and(masked_roi, masked_roi, mask=roi_mask)

        try:
            # Calculate GLCM for 4 directions (0, 45, 90, 135 degrees)
            # Distance = 1, Levels = 256
            glcm = graycomatrix(masked_roi, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                               levels=256, symmetric=True, normed=True)

            # Calculate homogeneity (also known as Inverse Difference Moment)
            homogeneity = graycoprops(glcm, 'homogeneity')

            # Average across all directions
            avg_homogeneity = np.mean(homogeneity)
            homogeneity_values.append(avg_homogeneity)

        except Exception as e:
            print(f"    Warning: Could not calculate GLCM for {object_type} {i+1}: {e}")
            continue

    if homogeneity_values:
        mean_homogeneity = np.mean(homogeneity_values)
        std_homogeneity = np.std(homogeneity_values)
        print(f"  {object_type} GLCM Homogeneity: {mean_homogeneity:.4f} ± {std_homogeneity:.4f}")
    else:
        mean_homogeneity = None
        std_homogeneity = None
        print(f"  {object_type}: Could not calculate GLCM homogeneity")

    return {
        'values': homogeneity_values,
        'mean': mean_homogeneity,
        'std': std_homogeneity
    }


def calculate_glcm_contrast(image, mask, contours, object_type="OC Structures"):
    """
    Calculate GLCM contrast for each object (OCS)
    Contrast measures the local variations in the GLCM
    Higher values indicate more contrast/heterogeneity
    """
    contrast_values = []

    for i, contour in enumerate(contours):
        # Get bounding box for the contour
        x, y, w, h = cv2.boundingRect(contour)

        # Extract ROI from image and mask
        roi = image[y:y+h, x:x+w]
        roi_mask = mask[y:y+h, x:x+w]

        # Create masked ROI
        masked_roi = cv2.bitwise_and(roi, roi, mask=roi_mask)

        if masked_roi.size == 0 or np.sum(roi_mask) == 0:
            continue

        # Convert to grayscale if needed
        if len(masked_roi.shape) == 3:
            masked_roi = cv2.cvtColor(masked_roi, cv2.COLOR_BGR2GRAY)

        # Get only the pixels inside the mask
        valid_pixels = masked_roi[roi_mask > 0]

        if len(valid_pixels) < 4:  # Need minimum pixels for GLCM
            continue

        # Normalize to 0-255 range
        min_val = np.min(valid_pixels)
        max_val = np.max(valid_pixels)
        if max_val > min_val:
            masked_roi = ((masked_roi - min_val) / (max_val - min_val) * 255).astype(np.uint8)

        # Apply mask again after normalization
        masked_roi = cv2.bitwise_and(masked_roi, masked_roi, mask=roi_mask)

        try:
            # Calculate GLCM for 4 directions
            glcm = graycomatrix(masked_roi, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                               levels=256, symmetric=True, normed=True)

            # Calculate contrast
            contrast = graycoprops(glcm, 'contrast')

            # Average across all directions
            avg_contrast = np.mean(contrast)
            contrast_values.append(avg_contrast)

        except Exception as e:
            print(f"    Warning: Could not calculate GLCM for {object_type} {i+1}: {e}")
            continue

    if contrast_values:
        mean_contrast = np.mean(contrast_values)
        std_contrast = np.std(contrast_values)
        print(f"  {object_type} GLCM Contrast: {mean_contrast:.4f} ± {std_contrast:.4f}")
    else:
        mean_contrast = None
        std_contrast = None
        print(f"  {object_type}: Could not calculate GLCM contrast")

    return {
        'values': contrast_values,
        'mean': mean_contrast,
        'std': std_contrast
    }


def calculate_fractal_dimension(contour, image_shape):
    """
    Calculate the fractal dimension of a contour using the box-counting method
    Measures boundary complexity - higher values indicate more complex boundaries
    """
    # Create binary image with just the contour
    binary = np.zeros(image_shape[:2], dtype=np.uint8)
    cv2.drawContours(binary, [contour], -1, 255, 1)

    # Get coordinates of contour points
    points = np.column_stack(np.where(binary > 0))

    if len(points) < 4:
        return None

    # Determine the range of box sizes
    min_dim = min(image_shape[:2])
    max_box_size = min_dim // 4
    min_box_size = 2

    # Generate box sizes (powers of 2)
    box_sizes = []
    size = min_box_size
    while size <= max_box_size:
        box_sizes.append(size)
        size *= 2

    if len(box_sizes) < 3:
        return None

    counts = []

    for box_size in box_sizes:
        # Count non-empty boxes
        # Discretize points to boxes
        boxes = set()
        for point in points:
            box_x = point[0] // box_size
            box_y = point[1] // box_size
            boxes.add((box_x, box_y))

        counts.append(len(boxes))

    # Calculate fractal dimension using linear regression in log-log space
    coeffs = np.polyfit(np.log(box_sizes), np.log(counts), 1)
    fractal_dim = -coeffs[0]  # Negative slope gives fractal dimension

    print(f"  Plasma Membrane Fractal Dimension: {fractal_dim:.4f}")
    print(f"    (1.0 = simple line, >1.5 = complex boundary)")

    return fractal_dim


def save_measurements_to_csv(measurements, csv_path):
    """
    Save measurements to a CSV file including clustering statistics
    """
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Write headers
        writer.writerow(['Measurement Type', 'Label', 'Area (pixels)'])

        # Write platelet total area
        writer.writerow(['Platelet Total', 'Platelet', f"{measurements['platelet_total_area']:.0f}"])
        writer.writerow([])  # Empty row for separation

        # Write dense granules
        writer.writerow(['Dense Granules', '', ''])
        for i, area in enumerate(measurements['dense_granules_areas'], 1):
            writer.writerow(['', f'{i}D', f'{area:.0f}'])
        writer.writerow(['Dense Granules Total Count', measurements['dense_granules_count'], ''])
        writer.writerow(['Dense Granules Total Area', '', f"{sum(measurements['dense_granules_areas']):.0f}" if measurements['dense_granules_areas'] else '0'])
        writer.writerow([])  # Empty row for separation

        # Write OC structures
        writer.writerow(['OC Structures', '', ''])
        for i, area in enumerate(measurements['ocs_areas'], 1):
            writer.writerow(['', f'{i}O', f'{area:.0f}'])
        writer.writerow(['OCS Total Count', measurements['ocs_count'], ''])
        writer.writerow(['OCS Total Area', '', f"{sum(measurements['ocs_areas']):.0f}" if measurements['ocs_areas'] else '0'])

        # Write summary statistics
        writer.writerow([])
        writer.writerow(['Summary Statistics', '', ''])
        if measurements['dense_granules_areas']:
            writer.writerow(['Dense Granules Mean Area', '', f"{np.mean(measurements['dense_granules_areas']):.0f}"])
            writer.writerow(['Dense Granules Std Dev', '', f"{np.std(measurements['dense_granules_areas']):.0f}"])
        if measurements['ocs_areas']:
            writer.writerow(['OCS Mean Area', '', f"{np.mean(measurements['ocs_areas']):.0f}"])
            writer.writerow(['OCS Std Dev', '', f"{np.std(measurements['ocs_areas']):.0f}"])

        # Write clustering analysis
        writer.writerow([])
        writer.writerow(['CLUSTERING ANALYSIS', '', ''])
        writer.writerow(['Metric', 'Dense Granules', 'OC Structures'])

        dg_stats = measurements['dense_granules_clustering']
        ocs_stats = measurements['ocs_clustering']

        writer.writerow(['Object Count', dg_stats['count'], ocs_stats['count']])

        if dg_stats['mean'] is not None:
            writer.writerow(['Mean Distance (pixels)', f"{dg_stats['mean']:.2f}",
                           f"{ocs_stats['mean']:.2f}" if ocs_stats['mean'] is not None else 'N/A'])
            writer.writerow(['Median Distance (pixels)', f"{dg_stats['median']:.2f}",
                           f"{ocs_stats['median']:.2f}" if ocs_stats['median'] is not None else 'N/A'])
            writer.writerow(['Std Dev Distance (pixels)', f"{dg_stats['std']:.2f}",
                           f"{ocs_stats['std']:.2f}" if ocs_stats['std'] is not None else 'N/A'])
            writer.writerow(['Min Distance (pixels)', f"{dg_stats['min']:.2f}",
                           f"{ocs_stats['min']:.2f}" if ocs_stats['min'] is not None else 'N/A'])
            writer.writerow(['Max Distance (pixels)', f"{dg_stats['max']:.2f}",
                           f"{ocs_stats['max']:.2f}" if ocs_stats['max'] is not None else 'N/A'])
            writer.writerow(['Q1 (25%) Distance (pixels)', f"{dg_stats['q1']:.2f}",
                           f"{ocs_stats['q1']:.2f}" if ocs_stats['q1'] is not None else 'N/A'])
            writer.writerow(['Q3 (75%) Distance (pixels)', f"{dg_stats['q3']:.2f}",
                           f"{ocs_stats['q3']:.2f}" if ocs_stats['q3'] is not None else 'N/A'])
            writer.writerow(['Clustering Index', f"{dg_stats['clustering_index']:.3f}",
                           f"{ocs_stats['clustering_index']:.3f}" if ocs_stats['clustering_index'] is not None else 'N/A'])
        else:
            writer.writerow(['', 'Insufficient objects (<2)',
                           'Insufficient objects (<2)' if ocs_stats['mean'] is None else 'See OCS column'])

        # Add interpretation note
        writer.writerow([])
        writer.writerow(['Note:', 'Clustering Index = StdDev/Mean', ''])
        writer.writerow(['', 'Lower values (~0.5) = uniform spacing', ''])
        writer.writerow(['', 'Higher values (>1.0) = clustered distribution', ''])

        # Write texture analysis
        writer.writerow([])
        writer.writerow(['TEXTURE ANALYSIS', '', ''])
        writer.writerow(['Metric', 'Value', 'Interpretation'])

        # Plasma membrane fractal dimension
        if measurements['plasma_fractal_dimension'] is not None:
            fd = measurements['plasma_fractal_dimension']
            interpretation = "Simple" if fd < 1.2 else "Moderate" if fd < 1.5 else "Complex"
            writer.writerow(['Plasma Membrane Fractal Dimension', f"{fd:.4f}", interpretation])
        else:
            writer.writerow(['Plasma Membrane Fractal Dimension', 'N/A', ''])

        # Dense granules homogeneity
        dg_homo = measurements['dense_granules_homogeneity']
        if dg_homo['mean'] is not None:
            interpretation = "Heterogeneous" if dg_homo['mean'] < 0.5 else "Moderate" if dg_homo['mean'] < 0.8 else "Homogeneous"
            writer.writerow(['Dense Granules GLCM Homogeneity (mean)', f"{dg_homo['mean']:.4f}", interpretation])
            writer.writerow(['Dense Granules GLCM Homogeneity (std)', f"{dg_homo['std']:.4f}", ''])
            # Individual values
            for i, val in enumerate(dg_homo['values'], 1):
                writer.writerow([f'  {i}D Homogeneity', f"{val:.4f}", ''])
        else:
            writer.writerow(['Dense Granules GLCM Homogeneity', 'N/A', ''])

        # OCS contrast
        ocs_cont = measurements['ocs_contrast']
        if ocs_cont['mean'] is not None:
            interpretation = "Low contrast" if ocs_cont['mean'] < 50 else "Moderate" if ocs_cont['mean'] < 150 else "High contrast"
            writer.writerow(['OCS GLCM Contrast (mean)', f"{ocs_cont['mean']:.4f}", interpretation])
            writer.writerow(['OCS GLCM Contrast (std)', f"{ocs_cont['std']:.4f}", ''])
            # Individual values
            for i, val in enumerate(ocs_cont['values'], 1):
                writer.writerow([f'  {i}O Contrast', f"{val:.4f}", ''])
        else:
            writer.writerow(['OCS GLCM Contrast', 'N/A', ''])

        # Add texture interpretation notes
        writer.writerow([])
        writer.writerow(['Texture Notes:', '', ''])
        writer.writerow(['', 'Fractal Dimension: 1.0-1.2 (simple), 1.2-1.5 (moderate), >1.5 (complex)', ''])
        writer.writerow(['', 'GLCM Homogeneity: 0-0.5 (heterogeneous), 0.5-0.8 (moderate), >0.8 (homogeneous)', ''])
        writer.writerow(['', 'GLCM Contrast: <50 (low), 50-150 (moderate), >150 (high)', ''])

    print(f"Measurements saved to: {csv_path}")


def visualize_results(results, window_size=(1400, 900)):
    """
    Display the results in windows
    """
    # Show labeled composite mask
    cv2.namedWindow("Labeled Composite Mask", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Labeled Composite Mask", window_size[0], window_size[1])
    cv2.imshow("Labeled Composite Mask", results['labeled_composite'])

    # Show composite mask (without labels)
    cv2.namedWindow("Composite Color Mask", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Composite Color Mask", window_size[0], window_size[1])
    cv2.imshow("Composite Color Mask", results['composite_mask'])

    # Show final overlay with all contours
    cv2.namedWindow("Final Overlay", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Final Overlay", window_size[0], window_size[1])
    cv2.imshow("Final Overlay", results['final_overlay'])

    # Print measurements
    print("\n=== MEASUREMENTS ===")
    print(f"Platelet Total Area: {results['measurements']['platelet_total_area']:.0f} pixels")
    print(f"\nDense Granules ({results['measurements']['dense_granules_count']} total):")
    for i, area in enumerate(results['measurements']['dense_granules_areas'], 1):
        print(f"  {i}D: {area:.0f} pixels")
    print(f"\nOC Structures ({results['measurements']['ocs_count']} total):")
    for i, area in enumerate(results['measurements']['ocs_areas'], 1):
        print(f"  {i}O: {area:.0f} pixels")

    # Print clustering metrics
    print("\n=== CLUSTERING ANALYSIS ===")
    dg_stats = results['measurements']['dense_granules_clustering']
    ocs_stats = results['measurements']['ocs_clustering']

    if dg_stats['mean'] is not None:
        print(f"\nDense Granules Clustering:")
        print(f"  Mean distance: {dg_stats['mean']:.2f} pixels")
        print(f"  Clustering Index: {dg_stats['clustering_index']:.3f}")
    else:
        print(f"\nDense Granules: Insufficient objects for clustering analysis")

    if ocs_stats['mean'] is not None:
        print(f"\nOC Structures Clustering:")
        print(f"  Mean distance: {ocs_stats['mean']:.2f} pixels")
        print(f"  Clustering Index: {ocs_stats['clustering_index']:.3f}")
    else:
        print(f"\nOC Structures: Insufficient objects for clustering analysis")

    print("\nPress any key to close windows...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Automated platelet detection and quantitative feature extraction."
    )
    parser.add_argument("image_path", help="Path to the input EM image.")
    parser.add_argument("-o", "--output-dir", default="output",
                        help="Directory to save masks, overlays, CSV, and summary.")
    parser.add_argument("--plasma-threshold", type=float, default=135)
    parser.add_argument("--plasma-min-area", type=float, default=1000000)
    # Note: the working plasma max-area from the batch pipeline is 8e7; the
    # module default mirrors the notebook, so it is set explicitly here.
    parser.add_argument("--plasma-max-area", type=float, default=80000000)
    parser.add_argument("--blue-threshold", type=float, default=70)
    parser.add_argument("--blue-min-area", type=float, default=10000)
    parser.add_argument("--blue-max-area", type=float, default=800000)
    parser.add_argument("--white-threshold", type=float, default=155)
    parser.add_argument("--red-min-area", type=float, default=7000)
    parser.add_argument("--red-max-area", type=float, default=1000000)
    args = parser.parse_args()

    results = detect_and_process_platelet(
        image_path=args.image_path,
        plasma_threshold=args.plasma_threshold,
        plasma_min_area=args.plasma_min_area,
        plasma_max_area=args.plasma_max_area,
        blue_threshold=args.blue_threshold,
        blue_min_area=args.blue_min_area,
        blue_max_area=args.blue_max_area,
        white_threshold=args.white_threshold,
        red_min_area=args.red_min_area,
        red_max_area=args.red_max_area,
        output_dir=args.output_dir,
    )

    if results:
        print("\n=== Analysis Complete ===")
        print(f"Dense granules found: {len(results['blue_contours'])}")
        print(f"OC structures found: {len(results['red_contours'])}")
    else:
        print("Analysis failed - no platelet detected")

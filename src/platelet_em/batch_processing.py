"""Excel-driven batch processing of a platelet EM dataset.

:func:`process_batch_platelets` applies the automated detection pipeline
(:func:`platelet_em.detection.detect_and_process_platelet`) across an entire
dataset using **per-case intensity thresholds** supplied in an Excel workbook.

Expected inputs:

* **Threshold workbook** (``.xlsx``) with one row per case and columns
  ``[case_name, plasma_threshold, blue_threshold, white_threshold]``.
* **Dataset folder** whose first-level subfolders are case names, each
  containing that case's EM images (``.jpg/.jpeg/.png``).

For every image the pipeline writes organized outputs into the output directory:

* ``00_plasma_outlines/`` ``01_plasma_masks/`` ``02_final_overlays/``
  ``03_labeled_composites/`` ``04_csv_measurements/`` ``05_summary_reports/``

plus a top-level ``BATCH_SUMMARY.txt``. The ``04_csv_measurements`` folder is the
input to :mod:`platelet_em.consolidate`, which builds the final feature table.
"""

import os

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import glob

try:  # Allow running both as a package module and as a standalone script.
    from .detection import detect_and_process_platelet, save_measurements_to_csv
except ImportError:  # pragma: no cover - fallback for direct execution
    from detection import detect_and_process_platelet, save_measurements_to_csv


def process_batch_platelets(xlsx_path, dataset_path, output_dir):
    """
    Batch process platelet images with thresholds from Excel file.

    Args:
        xlsx_path: Path to Excel file with columns [case_name, plasma_threshold, blue_threshold, white_threshold]
        dataset_path: Path to root dataset folder (thesis images)
        output_dir: Path to output directory

    Returns:
        Dictionary with processing summary
    """

    print("="*80)
    print("BATCH PLATELET ANALYSIS PIPELINE")
    print("="*80)

    # Read thresholds from Excel
    print(f"\n[1/4] Loading thresholds from: {xlsx_path}")
    try:
        df = pd.read_excel(xlsx_path)
        # Assume columns are: case_name, plasma_threshold, blue_threshold, white_threshold
        # Adjust column names if needed
        threshold_dict = {}
        for _, row in df.iterrows():
            case_name = str(row.iloc[0]).strip()  # First column
            threshold_dict[case_name] = {
                'plasma_threshold': float(row.iloc[1]),
                'blue_threshold': float(row.iloc[2]),
                'white_threshold': float(row.iloc[3])
            }
        print(f"   Loaded thresholds for {len(threshold_dict)} cases")
    except Exception as e:
        print(f"   ERROR: Could not read Excel file: {e}")
        return None

    # Create organized output directories
    print(f"\n[2/4] Setting up output directories in: {output_dir}")
    output_folders = {
        'plasma_outlines': os.path.join(output_dir, '00_plasma_outlines'),
        'plasma_masks': os.path.join(output_dir, '01_plasma_masks'),
        'final_overlays': os.path.join(output_dir, '02_final_overlays'),
        'labeled_composites': os.path.join(output_dir, '03_labeled_composites'),
        'csv_files': os.path.join(output_dir, '04_csv_measurements'),
        'summary_files': os.path.join(output_dir, '05_summary_reports')
    }

    for folder in output_folders.values():
        os.makedirs(folder, exist_ok=True)
    print(f"   Created {len(output_folders)} output folders")

    # Find all images in dataset
    print(f"\n[3/4] Scanning dataset: {dataset_path}")
    dataset_path = Path(dataset_path)

    # Find all image files (jpg, png, jpeg)
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    all_images = []
    for ext in image_extensions:
        all_images.extend(dataset_path.rglob(ext))

    print(f"   Found {len(all_images)} images")

    # Group images by case (1st subfolder)
    case_images = {}
    for img_path in all_images:
        # Get relative path from dataset root
        rel_path = img_path.relative_to(dataset_path)

        # First part of path is the case name
        if len(rel_path.parts) >= 1:
            case_name = rel_path.parts[0]
            if case_name not in case_images:
                case_images[case_name] = []
            case_images[case_name].append(img_path)

    print(f"   Organized into {len(case_images)} cases")
    for case_name, images in case_images.items():
        print(f"      - {case_name}: {len(images)} images")

    # Process all images
    print(f"\n[4/4] Processing images...")
    print("-"*80)

    results_summary = {
        'total_images': len(all_images),
        'processed': 0,
        'failed': 0,
        'skipped': 0,
        'cases_processed': {},
        'errors': []
    }

    for case_name, images in case_images.items():
        print(f"\n>>> Processing Case: {case_name}")

        # Get thresholds for this case
        if case_name not in threshold_dict:
            print(f"    WARNING: No thresholds found for case '{case_name}' in Excel file")
            print(f"    Skipping {len(images)} images in this case")
            results_summary['skipped'] += len(images)
            results_summary['errors'].append(f"Case '{case_name}': No thresholds in Excel")
            continue

        thresholds = threshold_dict[case_name]
        print(f"    Thresholds: plasma={thresholds['plasma_threshold']}, "
              f"blue={thresholds['blue_threshold']}, white={thresholds['white_threshold']}")

        case_results = {
            'processed': 0,
            'failed': 0
        }

        # Process each image in this case
        for i, img_path in enumerate(images, 1):
            img_name = img_path.name
            print(f"    [{i}/{len(images)}] Processing: {img_name}...", end=' ')

            try:
                # Run detection with case-specific thresholds
                results = detect_and_process_platelet(
                    image_path=str(img_path),
                    plasma_threshold=thresholds['plasma_threshold'],
                    plasma_min_area=1000000,
                    plasma_max_area=80000000,
                    blue_threshold=thresholds['blue_threshold'],
                    blue_min_area=10000,
                    blue_max_area=800000,
                    white_threshold=thresholds['white_threshold'],
                    red_min_area=7000,
                    red_max_area=1000000,
                    output_dir=None  # We'll save manually to organized folders
                )

                if results is None:
                    print("FAILED (no platelet detected)")
                    case_results['failed'] += 1
                    results_summary['failed'] += 1
                    results_summary['errors'].append(f"{case_name}/{img_name}: No platelet detected")
                    continue

                # Save outputs to organized folders
                base_name = f"{case_name}_{img_path.stem}"

                # Save plasma outline
                plasma_outline_only = np.ones_like(results['final_overlay']) * 255
                cv2.drawContours(plasma_outline_only, [results['plasma_contour']], -1, (0, 0, 0), 3)
                cv2.imwrite(
                    os.path.join(output_folders['plasma_outlines'], f"{base_name}_plasma_outline.png"),
                    plasma_outline_only
                )

                # Save plasma mask
                cv2.imwrite(
                    os.path.join(output_folders['plasma_masks'], f"{base_name}_plasma_mask.png"),
                    results['plasma_mask']
                )

                # Save final overlay
                cv2.imwrite(
                    os.path.join(output_folders['final_overlays'], f"{base_name}_final_overlay.png"),
                    results['final_overlay']
                )

                # Save labeled composite
                cv2.imwrite(
                    os.path.join(output_folders['labeled_composites'], f"{base_name}_labeled_composite.png"),
                    results['labeled_composite']
                )

                # Save CSV
                csv_path = os.path.join(output_folders['csv_files'], f"{base_name}_measurements.csv")
                save_measurements_to_csv(results['measurements'], csv_path)

                # Save summary
                summary_path = os.path.join(output_folders['summary_files'], f"{base_name}_summary.txt")
                save_summary_file(results, summary_path, img_name, thresholds)

                print(f"SUCCESS (DG:{len(results['blue_contours'])}, OCS:{len(results['red_contours'])})")
                case_results['processed'] += 1
                results_summary['processed'] += 1

            except Exception as e:
                print(f"ERROR: {str(e)}")
                case_results['failed'] += 1
                results_summary['failed'] += 1
                results_summary['errors'].append(f"{case_name}/{img_name}: {str(e)}")

        results_summary['cases_processed'][case_name] = case_results
        print(f"    Case Summary: {case_results['processed']} processed, {case_results['failed']} failed")

    # Final summary
    print("\n" + "="*80)
    print("BATCH PROCESSING COMPLETE")
    print("="*80)
    print(f"Total images found: {results_summary['total_images']}")
    print(f"Successfully processed: {results_summary['processed']}")
    print(f"Failed: {results_summary['failed']}")
    print(f"Skipped (no thresholds): {results_summary['skipped']}")
    print(f"\nOutputs saved to: {output_dir}")

    if results_summary['errors']:
        print(f"\n{len(results_summary['errors'])} errors occurred:")
        for error in results_summary['errors'][:10]:  # Show first 10 errors
            print(f"  - {error}")
        if len(results_summary['errors']) > 10:
            print(f"  ... and {len(results_summary['errors']) - 10} more errors")

    # Save batch summary
    batch_summary_path = os.path.join(output_dir, 'BATCH_SUMMARY.txt')
    with open(batch_summary_path, 'w') as f:
        f.write("BATCH PLATELET ANALYSIS SUMMARY\n")
        f.write("="*80 + "\n\n")
        f.write(f"Dataset: {dataset_path}\n")
        f.write(f"Thresholds file: {xlsx_path}\n")
        f.write(f"Output directory: {output_dir}\n\n")
        f.write(f"Total images: {results_summary['total_images']}\n")
        f.write(f"Processed: {results_summary['processed']}\n")
        f.write(f"Failed: {results_summary['failed']}\n")
        f.write(f"Skipped: {results_summary['skipped']}\n\n")

        f.write("Cases processed:\n")
        for case_name, case_results in results_summary['cases_processed'].items():
            f.write(f"  {case_name}:\n")
            f.write(f"    Processed: {case_results['processed']}\n")
            f.write(f"    Failed: {case_results['failed']}\n")

        if results_summary['errors']:
            f.write(f"\nErrors ({len(results_summary['errors'])}):\n")
            for error in results_summary['errors']:
                f.write(f"  - {error}\n")

    print(f"\nBatch summary saved to: {batch_summary_path}")

    return results_summary


def save_summary_file(results, summary_path, image_name, thresholds):
    """
    Save summary text file for a single image
    """
    measurements = results['measurements']

    with open(summary_path, 'w') as f:
        f.write("Platelet Analysis Summary\n")
        f.write("=" * 50 + "\n")
        f.write(f"Image: {image_name}\n")
        f.write(f"Platelet total area: {results['plasma_area']:.0f} pixels\n")
        f.write(f"Dense granules count: {len(results['blue_contours'])}\n")
        f.write(f"OC structures count: {len(results['red_contours'])}\n")

        f.write("\nDense Granules Areas:\n")
        for i, area in enumerate(results['blue_areas'], 1):
            f.write(f"  {i}D: {area:.0f} pixels\n")

        f.write("\nOC Structures Areas:\n")
        for i, area in enumerate(results['red_areas'], 1):
            f.write(f"  {i}O: {area:.0f} pixels\n")

        # Clustering information
        f.write("\n" + "=" * 50 + "\n")
        f.write("CLUSTERING ANALYSIS\n")
        f.write("=" * 50 + "\n")

        dg_stats = measurements['dense_granules_clustering']
        if dg_stats['mean'] is not None:
            f.write("\nDense Granules Clustering:\n")
            f.write(f"  Mean distance: {dg_stats['mean']:.2f} pixels\n")
            f.write(f"  Median distance: {dg_stats['median']:.2f} pixels\n")
            f.write(f"  Std deviation: {dg_stats['std']:.2f} pixels\n")
            f.write(f"  Min distance: {dg_stats['min']:.2f} pixels\n")
            f.write(f"  Max distance: {dg_stats['max']:.2f} pixels\n")
            f.write(f"  Clustering Index: {dg_stats['clustering_index']:.3f}\n")

        ocs_stats = measurements['ocs_clustering']
        if ocs_stats['mean'] is not None:
            f.write("\nOC Structures Clustering:\n")
            f.write(f"  Mean distance: {ocs_stats['mean']:.2f} pixels\n")
            f.write(f"  Median distance: {ocs_stats['median']:.2f} pixels\n")
            f.write(f"  Std deviation: {ocs_stats['std']:.2f} pixels\n")
            f.write(f"  Min distance: {ocs_stats['min']:.2f} pixels\n")
            f.write(f"  Max distance: {ocs_stats['max']:.2f} pixels\n")
            f.write(f"  Clustering Index: {ocs_stats['clustering_index']:.3f}\n")

        # Texture analysis
        f.write("\n" + "=" * 50 + "\n")
        f.write("TEXTURE ANALYSIS\n")
        f.write("=" * 50 + "\n")

        if measurements['plasma_fractal_dimension'] is not None:
            f.write(f"\nPlasma Membrane Boundary Complexity:\n")
            f.write(f"  Fractal Dimension: {measurements['plasma_fractal_dimension']:.4f}\n")
            complexity = ("Simple" if measurements['plasma_fractal_dimension'] < 1.2
                         else "Moderate" if measurements['plasma_fractal_dimension'] < 1.5
                         else "Complex")
            f.write(f"  Complexity Level: {complexity}\n")

        if measurements['dense_granules_homogeneity']['mean'] is not None:
            f.write(f"\nDense Granules Texture:\n")
            f.write(f"  GLCM Homogeneity: {measurements['dense_granules_homogeneity']['mean']:.4f} ± "
                   f"{measurements['dense_granules_homogeneity']['std']:.4f}\n")
            texture_type = ("Heterogeneous" if measurements['dense_granules_homogeneity']['mean'] < 0.5
                           else "Moderate" if measurements['dense_granules_homogeneity']['mean'] < 0.8
                           else "Homogeneous")
            f.write(f"  Texture Type: {texture_type}\n")

        if measurements['ocs_contrast']['mean'] is not None:
            f.write(f"\nOC Structures Texture:\n")
            f.write(f"  GLCM Contrast: {measurements['ocs_contrast']['mean']:.4f} ± "
                   f"{measurements['ocs_contrast']['std']:.4f}\n")
            contrast_level = ("Low" if measurements['ocs_contrast']['mean'] < 50
                            else "Moderate" if measurements['ocs_contrast']['mean'] < 150
                            else "High")
            f.write(f"  Contrast Level: {contrast_level}\n")

        f.write("\nParameters used:\n")
        f.write(f"  Plasma threshold: {thresholds['plasma_threshold']}\n")
        f.write(f"  Blue threshold: {thresholds['blue_threshold']}\n")
        f.write(f"  White threshold: {thresholds['white_threshold']}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Excel-driven batch platelet analysis over a dataset of case folders."
    )
    parser.add_argument("xlsx_path", help="Excel workbook with per-case thresholds "
                                          "[case_name, plasma_threshold, blue_threshold, white_threshold].")
    parser.add_argument("dataset_path", help="Root dataset folder containing case subfolders.")
    parser.add_argument("output_dir", help="Output directory for organized results.")
    args = parser.parse_args()

    summary = process_batch_platelets(
        xlsx_path=args.xlsx_path,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
    )

    if summary:
        print("\n✓ Batch processing completed successfully!")
    else:
        print("\n✗ Batch processing failed!")

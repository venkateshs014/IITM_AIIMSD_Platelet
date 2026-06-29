"""Unified command-line interface for the platelet_em Phase-1 pipeline.

Primary (automated) pipeline::

    # Automated detection + feature extraction on a single image
    platelet-em detect IMAGE.png --output-dir out/

    # Excel-driven batch processing of a dataset of case folders
    platelet-em process thresholds.xlsx dataset/ output_batch/

    # Consolidate per-image CSVs into one feature-table workbook
    platelet-em consolidate output_batch/04_csv_measurements/ features.xlsx

Interactive tools (threshold tuning / alternative workflow)::

    # Interactive plasma membrane + dense granules on a single image
    platelet-em annotate IMAGE.png --output-dir out/

    # Add OCS annotation on top of a dual-threshold result
    platelet-em ocs ANNOTATED.png OUT.png

    # Interactive batch pre-annotation over a folder
    platelet-em batch input_dir/ output_dir/

    # Rule-based activation analysis on an annotated image
    platelet-em analyze ANNOTATED.png --save figure.png

The same subcommands are available as ``python -m platelet_em.cli ...``.
"""

import argparse


def _cmd_detect(args):
    from .detection import detect_and_process_platelet
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
        print(f"\nDense granules: {len(results['blue_contours'])}, "
              f"OC structures: {len(results['red_contours'])}")
    else:
        print("Analysis failed - no platelet detected")


def _cmd_process(args):
    from .batch_processing import process_batch_platelets
    process_batch_platelets(args.xlsx_path, args.dataset_path, args.output_dir)


def _cmd_consolidate(args):
    from .consolidate import consolidate_measurements_main
    consolidate_measurements_main(args.csv_folder, args.output_excel)


def _cmd_annotate(args):
    from .plasma_membrane import analyze_platelet
    result = analyze_platelet(args.image_path, output_dir=args.output_dir)
    print(result)


def _cmd_ocs(args):
    from .open_canalicular import mark_oc_with_gui
    mark_oc_with_gui(args.image_path, output_folder=None, output_file_path=args.output_file_path)


def _cmd_batch(args):
    from .batch_annotate import run_batch
    run_batch(args.input_folder, args.output_folder, work_dir=args.work_dir)


def _cmd_analyze(args):
    from .feature_analysis import analyze_platelet_activation
    results = analyze_platelet_activation(args.image_path, visualization_save_path=args.save)
    print(f"\nFinal Classification: {results.get('classification', 'ERROR')}")
    print(f"Confidence: {results.get('confidence', 0):.2f}")


def build_parser():
    parser = argparse.ArgumentParser(
        prog="platelet-em",
        description="Automated ultrastructural analysis of platelet activation in EM images.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ---- Primary automated pipeline ----
    p_detect = sub.add_parser(
        "detect",
        help="Automated detection + feature extraction on a single image.",
    )
    p_detect.add_argument("image_path", help="Path to the input EM image.")
    p_detect.add_argument("-o", "--output-dir", default="output", help="Output directory.")
    p_detect.add_argument("--plasma-threshold", type=float, default=135)
    p_detect.add_argument("--plasma-min-area", type=float, default=1000000)
    p_detect.add_argument("--plasma-max-area", type=float, default=80000000)
    p_detect.add_argument("--blue-threshold", type=float, default=70)
    p_detect.add_argument("--blue-min-area", type=float, default=10000)
    p_detect.add_argument("--blue-max-area", type=float, default=800000)
    p_detect.add_argument("--white-threshold", type=float, default=155)
    p_detect.add_argument("--red-min-area", type=float, default=7000)
    p_detect.add_argument("--red-max-area", type=float, default=1000000)
    p_detect.set_defaults(func=_cmd_detect)

    p_process = sub.add_parser(
        "process",
        help="Excel-driven batch processing of a dataset of case folders.",
    )
    p_process.add_argument("xlsx_path", help="Excel workbook with per-case thresholds.")
    p_process.add_argument("dataset_path", help="Root dataset folder of case subfolders.")
    p_process.add_argument("output_dir", help="Output directory for organized results.")
    p_process.set_defaults(func=_cmd_process)

    p_consolidate = sub.add_parser(
        "consolidate",
        help="Merge per-image CSVs into one feature-table workbook.",
    )
    p_consolidate.add_argument("csv_folder", help="Folder of '*_measurements.csv' files.")
    p_consolidate.add_argument("output_excel", help="Path for the consolidated workbook.")
    p_consolidate.set_defaults(func=_cmd_consolidate)

    # ---- Interactive tools ----
    p_annotate = sub.add_parser(
        "annotate",
        help="Interactive plasma-membrane (green) + dense-granule (blue) segmentation.",
    )
    p_annotate.add_argument("image_path", help="Path to the input EM image.")
    p_annotate.add_argument("-o", "--output-dir", default=".", help="Output directory.")
    p_annotate.set_defaults(func=_cmd_annotate)

    p_ocs = sub.add_parser("ocs", help="Interactive Open Canalicular System (red) detection.")
    p_ocs.add_argument("image_path", help="Annotated image with a green plasma-membrane contour.")
    p_ocs.add_argument("output_file_path", help="Path to write the OCS-annotated result.")
    p_ocs.set_defaults(func=_cmd_ocs)

    p_batch = sub.add_parser("batch", help="Run the full Phase-1 pipeline over a folder.")
    p_batch.add_argument("input_folder", help="Folder of raw EM images.")
    p_batch.add_argument("output_folder", help="Folder for annotated outputs.")
    p_batch.add_argument("--work-dir", default=None, help="Folder for intermediate artifacts.")
    p_batch.set_defaults(func=_cmd_batch)

    p_analyze = sub.add_parser("analyze", help="Rule-based activation analysis on an annotated image.")
    p_analyze.add_argument("image_path", help="Path to the annotated platelet image.")
    p_analyze.add_argument(
        "-s", "--save", default="activation_analysis_results.png",
        help="Path to save the analysis visualization figure.",
    )
    p_analyze.set_defaults(func=_cmd_analyze)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

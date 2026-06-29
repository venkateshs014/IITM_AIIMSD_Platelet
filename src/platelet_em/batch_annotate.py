"""Batch pre-annotation driver.

Runs the full Phase-1 pre-annotation pipeline over every image in an input
folder:

1. :func:`platelet_em.plasma_membrane.analyze_platelet` for interactive plasma
   membrane (green) and dense-granule (blue) segmentation.
2. :func:`platelet_em.open_canalicular.mark_oc_with_gui` to add the Open
   Canalicular System (red) annotation on top of the dual-threshold result.

Each fully annotated image is written to the output folder under its original
file name, producing the composite ground-truth masks consumed by the
deep-learning stage (Phase 2).
"""

import os

try:  # Allow running both as a package module and as a standalone script.
    from .plasma_membrane import analyze_platelet
    from .open_canalicular import mark_oc_with_gui
except ImportError:  # pragma: no cover - fallback for direct execution
    from plasma_membrane import analyze_platelet
    from open_canalicular import mark_oc_with_gui


def run_batch(input_folder, output_folder, work_dir=None):
    """Annotate every image in ``input_folder`` and write results to ``output_folder``.

    Args:
        input_folder (str): Directory of raw EM images to annotate.
        output_folder (str): Directory where final annotated images are written
            (one per input image, keeping the original file name).
        work_dir (str, optional): Directory for intermediate artifacts produced
            by :func:`analyze_platelet`. Defaults to ``output_folder``.
    """
    work_dir = work_dir or output_folder
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(work_dir, exist_ok=True)

    for fname in os.listdir(input_folder):
        image_path = os.path.join(input_folder, fname)
        if not os.path.isfile(image_path):
            continue

        print(f"Processing {image_path}...")
        result = analyze_platelet(image_path, output_dir=work_dir)

        if result["status"] != "completed":
            print(f"skipped {image_path}: {result['message']}")
            continue

        # The dual-threshold (green + blue) result becomes the input to OCS marking.
        dual_threshold_file = result["output_files"]["final_result"]
        output_file_path = os.path.join(output_folder, fname)

        mark_oc_with_gui(dual_threshold_file, output_folder, output_file_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Batch Phase-1 pre-annotation (plasma membrane + dense granules + OCS)."
    )
    parser.add_argument("input_folder", help="Folder of raw EM images to annotate.")
    parser.add_argument("output_folder", help="Folder where annotated images are written.")
    parser.add_argument(
        "--work-dir", default=None,
        help="Folder for intermediate artifacts (default: output_folder).",
    )
    args = parser.parse_args()

    run_batch(args.input_folder, args.output_folder, work_dir=args.work_dir)

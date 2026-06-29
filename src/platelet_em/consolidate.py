"""Consolidate per-image measurement CSVs into a single feature-table workbook.

After :mod:`platelet_em.batch_processing` writes one ``*_measurements.csv`` per
image, :func:`consolidate_csv_to_excel` parses those files and assembles a single
Excel workbook with **one sheet per case**. Each row corresponds to one image and
columns hold the extracted morphological descriptors:

* Platelet total area, plasma-membrane fractal dimension
* Dense-granule count / total area / clustering index / GLCM homogeneity
* OCS count / total area / clustering index / GLCM contrast

This consolidated table is the tabular feature input used for activation grading.

Two small parsers locate values inside the (semi-structured) CSVs:
:func:`find_value_by_label` (search a label and read an offset cell) and
:func:`get_fixed_cell_value` (read an Excel-style cell reference such as ``C2``).
"""

import os
from pathlib import Path
from collections import defaultdict

import pandas as pd
import openpyxl


def find_value_by_label(csv_path, search_label, offset_cols=1, occurrence=1):
    """
    Find a label in CSV and return the value at specified column offset.

    Args:
        csv_path: Path to CSV file
        search_label: Text to search for
        offset_cols: Number of columns to the right (1 = immediate right, 2 = 2nd cell right)
        occurrence: Which occurrence to use if label appears multiple times (1-based)

    Returns:
        Value or None if not found
    """
    try:
        df = pd.read_csv(csv_path, header=None)

        # Search for the label in all cells
        found_count = 0
        for row_idx in range(len(df)):
            for col_idx in range(len(df.columns)):
                cell_value = str(df.iloc[row_idx, col_idx]).strip()

                # Check if this cell contains the search label
                if search_label.lower() in cell_value.lower():
                    found_count += 1

                    # Check if this is the occurrence we want
                    if found_count == occurrence:
                        # Get value at offset
                        target_col = col_idx + offset_cols
                        if target_col < len(df.columns):
                            value = df.iloc[row_idx, target_col]
                            # Handle NaN or empty values
                            if pd.isna(value) or value == '' or value == 'N/A':
                                return None
                            return value
                        return None

        return None
    except Exception as e:
        print(f"    Warning: Error searching '{search_label}' in {os.path.basename(csv_path)}: {e}")
        return None


def get_fixed_cell_value(csv_path, cell_ref):
    """
    Get value from a specific Excel-style cell reference (e.g., 'C2').

    Args:
        csv_path: Path to CSV file
        cell_ref: Excel-style cell reference (e.g., 'C2', 'B25')

    Returns:
        Value or None if not found
    """
    try:
        # Convert Excel cell reference to 0-based indices
        col_letter = ''.join([c for c in cell_ref if c.isalpha()])
        row_num = int(''.join([c for c in cell_ref if c.isdigit()]))

        # Convert column letter to index (A=0, B=1, C=2, etc.)
        col_idx = 0
        for char in col_letter.upper():
            col_idx = col_idx * 26 + (ord(char) - ord('A') + 1)
        col_idx -= 1  # Convert to 0-based

        row_idx = row_num - 1  # Convert to 0-based

        df = pd.read_csv(csv_path, header=None)

        if row_idx < len(df) and col_idx < len(df.columns):
            value = df.iloc[row_idx, col_idx]
            if pd.isna(value) or value == '' or value == 'N/A':
                return None
            return value
        return None
    except Exception as e:
        print(f"    Warning: Error reading cell {cell_ref} from {os.path.basename(csv_path)}: {e}")
        return None


def consolidate_csv_to_excel(csv_folder_path, output_excel_path):
    """
    Consolidate all CSV measurement files into a single Excel file with one sheet per case.

    Args:
        csv_folder_path: Path to folder containing CSV measurement files
        output_excel_path: Path where consolidated Excel file will be saved

    Returns:
        Dictionary with consolidation summary
    """

    print("="*80)
    print("CONSOLIDATING CSV MEASUREMENTS TO EXCEL")
    print("="*80)

    csv_folder = Path(csv_folder_path)

    # Find all CSV files
    csv_files = list(csv_folder.glob("*_measurements.csv"))

    if not csv_files:
        print(f"ERROR: No CSV files found in {csv_folder_path}")
        return None

    print(f"\nFound {len(csv_files)} CSV files")

    # Group CSV files by case name
    case_data = defaultdict(list)

    for csv_file in csv_files:
        # Extract case name from filename
        filename = csv_file.stem  # Remove .csv extension
        parts = filename.split('_')

        # Find where "measurements" is and take everything before it
        if 'measurements' in parts:
            case_parts = parts[:parts.index('measurements')]
        else:
            case_parts = parts[:-1]

        # Try to identify case name (usually first 1-2 parts)
        if len(case_parts) >= 2 and case_parts[0].lower().startswith('case'):
            case_name = f"{case_parts[0]}_{case_parts[1]}"
        else:
            case_name = case_parts[0] if case_parts else "Unknown"

        # Extract image name
        if len(case_parts) > 2:
            image_name = '_'.join(case_parts[2:])
        else:
            image_name = filename.replace('_measurements', '')

        case_data[case_name].append({
            'csv_path': csv_file,
            'image_name': image_name,
            'original_filename': csv_file.name
        })

    print(f"\nGrouped into {len(case_data)} cases:")
    for case_name, files in case_data.items():
        print(f"  - {case_name}: {len(files)} images")

    # Define column extraction rules
    # Format: (header_name, extraction_function)
    column_definitions = [
        ('Image Name', 'image_name'),
        ('Platelet Total Area', ('fixed_cell', 'C2')),
        ('Plasma Fractal Dimension', ('search', 'Plasma Membrane Fractal Dimension', 1, 1)),
        ('Dense Granules Count', ('search', 'Dense Granules Total Count', 1, 1)),
        ('Dense Granules Total Area', ('search', 'Dense Granules Total Area', 2, 1)),
        ('Dense Granules Clustering Index', ('search', 'Clustering Index', 1, 1)),  # First occurrence
        ('Dense Granules Homogeneity', ('search', 'Dense Granules GLCM Homogeneity (mean)', 1, 1)),
        ('OCS Count', ('search', 'OCS Total Count', 1, 1)),
        ('OCS Total Area', ('search', 'OCS Total Area', 2, 1)),
        ('OCS Clustering Index', ('search', 'Clustering Index', 2, 2)),  # Second occurrence, 2 cells right
        ('OCS Contrast', ('search', 'OCS GLCM Contrast (mean)', 1, 1))
    ]

    # Create Excel writer
    print(f"\nCreating Excel file: {output_excel_path}")

    with pd.ExcelWriter(output_excel_path, engine='openpyxl') as writer:

        summary = {
            'total_cases': len(case_data),
            'total_images': 0,
            'cases_processed': {}
        }

        for case_name, images_data in sorted(case_data.items()):
            print(f"\nProcessing case: {case_name} ({len(images_data)} images)")

            # Prepare data for this case
            rows = []

            for img_data in images_data:
                csv_path = img_data['csv_path']
                image_name = img_data['image_name']

                print(f"  Processing: {image_name}...", end=' ')

                # Extract values based on column definitions
                row_data = []

                for col_name, extraction_rule in column_definitions:
                    if extraction_rule == 'image_name':
                        # Just use the image name
                        value = image_name
                    elif isinstance(extraction_rule, tuple):
                        if extraction_rule[0] == 'fixed_cell':
                            # Extract from fixed cell reference
                            cell_ref = extraction_rule[1]
                            value = get_fixed_cell_value(csv_path, cell_ref)
                        elif extraction_rule[0] == 'search':
                            # Search for label and extract value
                            search_label = extraction_rule[1]
                            offset = extraction_rule[2] if len(extraction_rule) > 2 else 1
                            occurrence = extraction_rule[3] if len(extraction_rule) > 3 else 1
                            value = find_value_by_label(csv_path, search_label, offset, occurrence)
                        else:
                            value = None
                    else:
                        value = None

                    # Try to convert to float if possible
                    if value is not None and value != '':
                        try:
                            value = float(value)
                        except (ValueError, TypeError):
                            pass  # Keep as string

                    row_data.append(value)

                rows.append(row_data)
                print("✓")

            # Create DataFrame
            headers = [col[0] for col in column_definitions]
            df = pd.DataFrame(rows, columns=headers)

            df = df.sort_values('Image Name').reset_index(drop=True)

            # Clean sheet name (Excel sheet names have restrictions)
            sheet_name = case_name.replace('/', '_').replace('\\', '_')[:31]  # Max 31 chars

            # Write to Excel
            df.to_excel(writer, sheet_name=sheet_name, index=False)

            # Format the worksheet
            worksheet = writer.sheets[sheet_name]

            # Auto-adjust column widths
            for idx, col in enumerate(df.columns, 1):
                max_length = max(
                    df[col].astype(str).apply(len).max(),
                    len(col)
                ) + 2
                worksheet.column_dimensions[openpyxl.utils.get_column_letter(idx)].width = min(max_length, 50)

            # Bold header row
            for cell in worksheet[1]:
                cell.font = openpyxl.styles.Font(bold=True)

            # Freeze first row
            worksheet.freeze_panes = 'A2'

            print(f"  ✓ Added sheet '{sheet_name}' with {len(rows)} rows")

            summary['cases_processed'][case_name] = len(rows)
            summary['total_images'] += len(rows)

    print("\n" + "="*80)
    print("CONSOLIDATION COMPLETE")
    print("="*80)
    print(f"Total cases processed: {summary['total_cases']}")
    print(f"Total images consolidated: {summary['total_images']}")
    print(f"Output file: {output_excel_path}")

    # Verify file was created
    if os.path.exists(output_excel_path):
        file_size = os.path.getsize(output_excel_path) / 1024  # KB
        print(f"File size: {file_size:.2f} KB")
        print("\n✓ Excel file created successfully!")
    else:
        print("\n✗ Error: Excel file was not created!")
        return None

    return summary


def consolidate_measurements_main(csv_folder_path, output_excel_path):
    """
    Main function to consolidate CSV measurements into Excel file.

    Args:
        csv_folder_path: Path to folder with CSV files (e.g., "output_batch/04_csv_measurements")
        output_excel_path: Path for output Excel file (e.g., "consolidated_measurements.xlsx")
    """

    # Validate inputs
    if not os.path.exists(csv_folder_path):
        print(f"ERROR: CSV folder not found: {csv_folder_path}")
        return None

    # Create output directory if needed
    output_dir = os.path.dirname(output_excel_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Run consolidation
    summary = consolidate_csv_to_excel(csv_folder_path, output_excel_path)

    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Consolidate per-image measurement CSVs into a single Excel feature table."
    )
    parser.add_argument("csv_folder", help="Folder of '*_measurements.csv' files "
                                           "(e.g., output_batch/04_csv_measurements).")
    parser.add_argument("output_excel", help="Path for the consolidated Excel workbook.")
    args = parser.parse_args()

    summary = consolidate_measurements_main(
        csv_folder_path=args.csv_folder,
        output_excel_path=args.output_excel,
    )

    if summary:
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        for case_name, count in summary['cases_processed'].items():
            print(f"{case_name}: {count} images")
        print("\n✓ All done! Check the Excel file for consolidated data.")
    else:
        print("\n✗ Consolidation failed!")

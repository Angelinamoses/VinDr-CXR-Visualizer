import os

import pandas as pd
import pydicom
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D


# =========================================================
# FILE PATHS
# =========================================================

ANNOTATIONS_PATH = "data/annotations/annotations_train.csv"
LABELS_PATH = "data/annotations/image_labels_train.csv"
IMAGE_FOLDER = "data/images"


# =========================================================
# LOAD DATA
# =========================================================

def load_data():
    """
    Load the training annotation and classification CSV files.
    """

    annotations = pd.read_csv(ANNOTATIONS_PATH)
    labels = pd.read_csv(LABELS_PATH)

    return annotations, labels


# =========================================================
# FIND DICOM FILE
# =========================================================

def find_dicom(image_id):
    """
    Find the DICOM file corresponding to an image ID.
    """

    dicom_path = os.path.join(
        IMAGE_FOLDER,
        f"{image_id}.dicom"
    )

    if not os.path.exists(dicom_path):
        raise FileNotFoundError(
            f"\nDICOM file not found:\n{dicom_path}"
        )

    return dicom_path


# =========================================================
# GET CLASSIFICATION FINDINGS
# =========================================================

def get_classification_labels(labels, image_id):
    """
    Get all classification findings reported by
    each radiologist for the selected image.
    """

    image_labels = labels[
        labels["image_id"] == image_id
    ]

    results = []

    for _, row in image_labels.iterrows():

        findings = []

        # First two columns are image_id and rad_id.
        # Remaining columns represent disease/findings.
        for column in labels.columns[2:]:

            if row[column] == 1:
                findings.append(column)

        results.append({
            "radiologist": row["rad_id"],
            "findings": findings
        })

    return results


# =========================================================
# GET BOUNDING-BOX ANNOTATIONS
# =========================================================

def get_annotations(annotations, image_id):
    """
    Get all annotation records for the selected image.
    """

    image_annotations = annotations[
        annotations["image_id"] == image_id
    ]

    return image_annotations


# =========================================================
# VISUALIZE IMAGE
# =========================================================

def visualize_image(image_id, annotations, labels):
    """
    Display the DICOM image together with:
    - Radiologist classification findings
    - Bounding-box annotations
    """

    # -----------------------------------------------------
    # Find DICOM
    # -----------------------------------------------------

    dicom_path = find_dicom(image_id)

    # -----------------------------------------------------
    # Read DICOM
    # -----------------------------------------------------

    ds = pydicom.dcmread(dicom_path)

    # Extract image pixel data
    image = ds.pixel_array

    # -----------------------------------------------------
    # Handle DICOM photometric interpretation
    # -----------------------------------------------------

    # Some DICOM images use MONOCHROME1, where lower
    # pixel values appear brighter. Invert for display.
    if ds.get("PhotometricInterpretation") == "MONOCHROME1":
        image = image.max() - image

    # -----------------------------------------------------
    # Get annotations and classifications
    # -----------------------------------------------------

    image_annotations = get_annotations(
        annotations,
        image_id
    )

    classifications = get_classification_labels(
        labels,
        image_id
    )

    # -----------------------------------------------------
    # Create figure
    # -----------------------------------------------------

    fig = plt.figure(
        figsize=(16, 10)
    )

    # -----------------------------------------------------
    # Create image and information areas
    # -----------------------------------------------------

    ax = fig.add_axes(
        [0.04, 0.08, 0.62, 0.82]
    )

    info_ax = fig.add_axes(
        [0.70, 0.08, 0.27, 0.82]
    )

    # -----------------------------------------------------
    # Display X-ray
    # -----------------------------------------------------

    ax.imshow(
        image,
        cmap="gray"
    )

    # -----------------------------------------------------
    # Bounding-box line styles
    # -----------------------------------------------------

    line_styles = {
        "R8": "-",
        "R9": "--",
        "R10": ":"
    }

    # -----------------------------------------------------
    # Draw bounding boxes
    # -----------------------------------------------------

    for _, row in image_annotations.iterrows():

        # Some annotations do not have bounding boxes.
        if pd.isna(row["x_min"]):
            continue

        x_min = row["x_min"]
        y_min = row["y_min"]

        width = row["x_max"] - row["x_min"]
        height = row["y_max"] - row["y_min"]

        rad_id = row["rad_id"]

        rectangle = Rectangle(
            (x_min, y_min),
            width,
            height,
            fill=False,
            linewidth=2,
            linestyle=line_styles.get(
                rad_id,
                "-"
            )
        )

        ax.add_patch(rectangle)

    # -----------------------------------------------------
    # Image title
    # -----------------------------------------------------

    ax.set_title(
        f"Image ID: {image_id}",
        fontsize=14,
        fontweight="bold",
        pad=10
    )

    ax.axis("off")

    # -----------------------------------------------------
    # Annotation legend
    # -----------------------------------------------------

    legend_handles = [
        Line2D(
            [0],
            [0],
            linestyle="-",
            linewidth=2,
            label="R8"
        ),
        Line2D(
            [0],
            [0],
            linestyle="--",
            linewidth=2,
            label="R9"
        ),
        Line2D(
            [0],
            [0],
            linestyle=":",
            linewidth=2,
            label="R10"
        )
    ]

    ax.legend(
        handles=legend_handles,
        loc="lower right",
        title="Radiologist",
        framealpha=0.9
    )

    # =====================================================
    # INFORMATION PANEL
    # =====================================================

    info_ax.axis("off")

    information = []

    # -----------------------------------------------------
    # Image information
    # -----------------------------------------------------

    information.append(
        "IMAGE INFORMATION\n"
    )

    information.append(
        "────────────────────────────\n"
    )

    information.append(
        f"Image ID:\n{image_id}\n\n"
    )

    # -----------------------------------------------------
    # Radiologist classifications
    # -----------------------------------------------------

    information.append(
        "RADIOLOGIST CLASSIFICATIONS\n"
    )

    information.append(
        "────────────────────────────\n"
    )

    if not classifications:

        information.append(
            "No classification records found.\n\n"
        )

    else:

        for result in classifications:

            information.append(
                f"{result['radiologist']}\n"
            )

            if result["findings"]:

                for finding in result["findings"]:

                    information.append(
                        f"  • {finding}\n"
                    )

            else:

                information.append(
                    "  • No finding\n"
                )

            information.append("\n")

    # -----------------------------------------------------
    # Bounding-box information
    # -----------------------------------------------------

    information.append(
        "BOUNDING BOXES\n"
    )

    information.append(
        "────────────────────────────\n"
    )

    box_count = 0

    for _, row in image_annotations.iterrows():

        if pd.isna(row["x_min"]):
            continue

        box_count += 1

        information.append(
            f"{box_count}. "
            f"{row['rad_id']} → "
            f"{row['class_name']}\n"
        )

    if box_count == 0:

        information.append(
            "No bounding boxes available.\n"
        )

    else:

        information.append(
            f"\nTotal bounding boxes: {box_count}\n"
        )

    # -----------------------------------------------------
    # Display information panel
    # -----------------------------------------------------

    info_ax.text(
        0,
        1,
        "".join(information),
        fontsize=10.5,
        verticalalignment="top",
        family="monospace"
    )

    # =====================================================
    # OVERALL TITLE
    # =====================================================

    fig.suptitle(
        "VinDr-CXR Annotation Viewer",
        fontsize=20,
        fontweight="bold",
        y=0.96
    )

    # -----------------------------------------------------
    # IMPORTANT: Display the figure
    # -----------------------------------------------------

    plt.show()


# =========================================================
# MAIN PROGRAM
# =========================================================

def main():

    print("\n" + "=" * 60)
    print("VinDr-CXR Annotation Viewer")
    print("=" * 60)

    # -----------------------------------------------------
    # Load CSV data
    # -----------------------------------------------------

    annotations, labels = load_data()

    # -----------------------------------------------------
    # Ask user for image ID
    # -----------------------------------------------------

    image_id = input(
        "\nEnter VinDr-CXR Image ID: "
    ).strip()

    # -----------------------------------------------------
    # Validate image ID
    # -----------------------------------------------------

    if image_id == "":
        print("\nError: Image ID cannot be empty.")
        return

    # -----------------------------------------------------
    # Check whether the image exists
    # -----------------------------------------------------

    try:

        visualize_image(
            image_id,
            annotations,
            labels
        )

    except FileNotFoundError as error:

        print(error)

    except Exception as error:

        print(
            f"\nAn unexpected error occurred:\n{error}"
        )


# =========================================================
# PROGRAM ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()
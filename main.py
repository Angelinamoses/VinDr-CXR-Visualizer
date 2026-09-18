import os

import pandas as pd
import pydicom
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


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

    annotations = pd.read_csv(ANNOTATIONS_PATH)
    labels = pd.read_csv(LABELS_PATH)

    return annotations, labels


# =========================================================
# FIND DICOM FILE
# =========================================================

def find_dicom(image_id):

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

    image_labels = labels[
        labels["image_id"] == image_id
    ]

    results = []

    for _, row in image_labels.iterrows():

        findings = []

        for column in labels.columns[2:]:

            if row[column] == 1:
                findings.append(column)

        results.append({
            "radiologist": row["rad_id"],
            "findings": findings
        })

    return results


# =========================================================
# GET BOUNDING BOX ANNOTATIONS
# =========================================================

def get_annotations(annotations, image_id):

    image_annotations = annotations[
        annotations["image_id"] == image_id
    ]

    return image_annotations


# =========================================================
# VISUALIZE IMAGE
# =========================================================

def visualize_image(image_id, annotations, labels):

    # -----------------------------------------------------
    # Find DICOM
    # -----------------------------------------------------

    dicom_path = find_dicom(image_id)

    # -----------------------------------------------------
    # Read DICOM
    # -----------------------------------------------------

    ds = pydicom.dcmread(dicom_path)

    image = ds.pixel_array

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

    # Image area
    ax = fig.add_axes(
        [0.05, 0.08, 0.62, 0.82]
    )

    # Information area
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
    # Draw bounding boxes
    # -----------------------------------------------------

    # Different line styles for different radiologists
    line_styles = {
        "R8": "-",
        "R9": "--",
        "R10": ":"
    }

    for _, row in image_annotations.iterrows():

        # Skip annotations without bounding boxes
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
        plt.Line2D(
            [0], [0],
            linestyle="-",
            linewidth=2,
            label="R8"
        ),
        plt.Line2D(
            [0], [0],
            linestyle="--",
            linewidth=2,
            label="R9"
        ),
        plt.Line2D(
            [0], [0],
            linestyle=":",
            linewidth=2,
            label="R10"
        )
    ]

    ax.legend(
        handles=legend_handles,
        loc="lower right",
        title="Radiologist"
    )

    # -----------------------------------------------------
    # Information panel
    # -----------------------------------------------------

    info_ax.axis("off")

    information = []

    information.append(
        "IMAGE INFORMATION\n"
        "────────────────────────\n"
    )

    information.append(
        f"Image ID:\n{image_id}\n"
    )

    information.append(
        "RADIOLOGIST CLASSIFICATIONS\n"
        "────────────────────────\n"
    )

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

    information.append(
        "BOUNDING BOXES\n"
        "────────────────────────\n"
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

    information.append(
        f"\nTotal bounding boxes: {box_count}"
    )

    info_ax.text(
        0,
        1,
        "".join(information),
        fontsize=11,
        verticalalignment="top",
        family="monospace"
    )

    # -----------------------------------------------------
    # Overall figure title
    # -----------------------------------------------------

    fig.suptitle(
    "VinDr-CXR Annotation Viewer",
    fontsize=20,
    fontweight="bold",
    y=0.96
)


# =========================================================
# MAIN PROGRAM
# =========================================================

def main():

    print("\n" + "=" * 60)
    print("VinDr-CXR Annotation Viewer")
    print("=" * 60)

    annotations, labels = load_data()

    image_id = input(
        "\nEnter VinDr-CXR Image ID: "
    ).strip()

    visualize_image(
        image_id,
        annotations,
        labels
    )


# =========================================================
# PROGRAM ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()
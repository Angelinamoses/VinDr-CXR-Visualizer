import pandas as pd


ANNOTATIONS_PATH = "data/annotations/annotations_train.csv"
LABELS_PATH = "data/annotations/image_labels_train.csv"


def main():

    # Load CSV files
    annotations = pd.read_csv(ANNOTATIONS_PATH)
    labels = pd.read_csv(LABELS_PATH)

    # Ask user for image ID
    image_id = input("\nEnter Image ID: ").strip()

    # Find annotations for this image
    image_annotations = annotations[
        annotations["image_id"] == image_id
    ]

    # Find classification labels for this image
    image_labels = labels[
        labels["image_id"] == image_id
    ]

    print("\n" + "=" * 70)
    print("IMAGE INFORMATION")
    print("=" * 70)

    print("Image ID:", image_id)

    # --------------------------------------------------
    # Bounding-box annotations
    # --------------------------------------------------

    print("\n" + "-" * 70)
    print("BOUNDING-BOX ANNOTATIONS")
    print("-" * 70)

    if image_annotations.empty:
        print("No annotations found.")
    else:
        for _, row in image_annotations.iterrows():

            print(f"\nRadiologist: {row['rad_id']}")
            print(f"Finding: {row['class_name']}")

            if pd.isna(row["x_min"]):
                print("Bounding box: Not available")
            else:
                print(
                    f"Bounding box: "
                    f"({row['x_min']:.2f}, {row['y_min']:.2f}) → "
                    f"({row['x_max']:.2f}, {row['y_max']:.2f})"
                )

    # --------------------------------------------------
    # Classification labels
    # --------------------------------------------------

    print("\n" + "-" * 70)
    print("CLASSIFICATION LABELS")
    print("-" * 70)

    if image_labels.empty:
        print("No classification labels found.")
    else:

        for _, row in image_labels.iterrows():

            print(f"\nRadiologist: {row['rad_id']}")

            findings = []

            for column in labels.columns[2:]:

                if row[column] == 1:
                    findings.append(column)

            if findings:
                print("Findings:", ", ".join(findings))
            else:
                print("Findings: None")


if __name__ == "__main__":
    main()
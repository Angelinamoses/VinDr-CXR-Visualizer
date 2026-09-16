import pandas as pd


ANNOTATIONS_PATH = "data/annotations/annotations_train.csv"
LABELS_PATH = "data/annotations/image_labels_train.csv"


def main():

    annotations = pd.read_csv(ANNOTATIONS_PATH)
    labels = pd.read_csv(LABELS_PATH)

    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)

    print(f"Annotation rows: {len(annotations):,}")
    print(f"Classification rows: {len(labels):,}")
    print(f"Unique images: {labels['image_id'].nunique():,}")

    # ---------------------------------------------------------
    # Find annotations that actually have bounding boxes
    # ---------------------------------------------------------

    boxes = annotations.dropna(
        subset=["x_min", "y_min", "x_max", "y_max"]
    )

    print(f"Rows with bounding boxes: {len(boxes):,}")
    print(f"Unique images with bounding boxes: "
          f"{boxes['image_id'].nunique():,}")

    # ---------------------------------------------------------
    # Pick the first image that has a bounding box
    # ---------------------------------------------------------

    image_id = boxes.iloc[0]["image_id"]

    print("\n" + "=" * 60)
    print("SELECTED IMAGE")
    print("=" * 60)

    print("Image ID:", image_id)

    # ---------------------------------------------------------
    # Show bounding-box annotations
    # ---------------------------------------------------------

    image_annotations = annotations[
        annotations["image_id"] == image_id
    ]

    print("\nBounding-box annotations:")
    print(image_annotations.to_string(index=False))

    # ---------------------------------------------------------
    # Show classification labels
    # ---------------------------------------------------------

    image_labels = labels[
        labels["image_id"] == image_id
    ]

    print("\nClassification labels:")
    print(image_labels.to_string(index=False))


if __name__ == "__main__":
    main()
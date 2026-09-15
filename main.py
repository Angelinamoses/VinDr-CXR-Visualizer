import pandas as pd


ANNOTATIONS_PATH = "data/annotations/annotations_train.csv"
LABELS_PATH = "data/annotations/image_labels_train.csv"


def main():

    # Load annotation data
    annotations = pd.read_csv(ANNOTATIONS_PATH)

    # Load image-level classification labels
    labels = pd.read_csv(LABELS_PATH)

    print("\n" + "=" * 60)
    print("ANNOTATIONS TRAIN")
    print("=" * 60)

    print("Shape:", annotations.shape)

    print("\nColumns:")
    print(annotations.columns.tolist())

    print("\nFirst 5 rows:")
    print(annotations.head())

    print("\n" + "=" * 60)
    print("IMAGE LABELS TRAIN")
    print("=" * 60)

    print("Shape:", labels.shape)

    print("\nColumns:")
    print(labels.columns.tolist())

    print("\nFirst 5 rows:")
    print(labels.head())


if __name__ == "__main__":
    main()
from pathlib import Path

import h5py
import hdf5plugin
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Kaggle data location.
DATA_DIR = Path(
    "/home/ibab/.cache/kagglehub/competitions/open-problems-multimodal"
)


def decode_labels(values):
    """
    Convert byte labels from HDF5 into normal text.
    """

    labels = []

    for value in values:
        labels.append(
            value.decode("utf-8").rstrip("\x00")
        )

    return labels


def inspect_multiome_input():
    """
    Inspect the Multiome ATAC input matrix.
    """

    file_path = DATA_DIR / "train_multi_inputs.h5"

    with h5py.File(file_path, "r") as file:

        data = file["train_multi_inputs"]

        # This is the ATAC matrix.
        values = data["block0_values"]

        # ATAC feature names are genomic regions.
        feature_names = decode_labels(
            data["axis0"][:]
        )

        # These are the cell IDs.
        cell_ids = decode_labels(
            data["axis1"][:]
        )

        print("File:", file_path.name)
        print("Matrix shape:", values.shape)
        print("Data type:", values.dtype)

        print("\nNumber of ATAC features:")
        print(len(feature_names))

        print("First five ATAC features:")
        print(feature_names[:5])

        print("\nNumber of cells:")
        print(len(cell_ids))

        print("First five cell IDs:")
        print(cell_ids[:5])

        # Read only the first cell.
        first_cell_values = values[0, :]

        nonzero_columns = np.flatnonzero(
            first_cell_values
        )

        print("\nFirst cell ID:")
        print(cell_ids[0])

        print("\nNumber of nonzero ATAC regions:")
        print(len(nonzero_columns))

        print("\nFirst ten nonzero ATAC measurements:")

        for column_number in nonzero_columns[:10]:
            print(
                "Column:", column_number,
                "| Region:", feature_names[column_number],
                "| Value:", first_cell_values[column_number]
            )
def inspect_multiome_target():
    """
    Inspect the RNA target for the Multiome task.
    """

    file_path = DATA_DIR / "train_multi_targets.h5"

    with h5py.File(file_path, "r") as file:

        data = file["train_multi_targets"]

        # This is the RNA target matrix.
        values = data["block0_values"]

        # Target feature names are genes.
        gene_names = decode_labels(
            data["axis0"][:]
        )

        # These are the target cell IDs.
        cell_ids = decode_labels(
            data["axis1"][:]
        )

        print("\n\nTARGET FILE:", file_path.name)
        print("Target matrix shape:", values.shape)
        print("Target data type:", values.dtype)

        print("\nNumber of target genes:")
        print(len(gene_names))

        print("First five target genes:")
        print(gene_names[:5])

        print("\nNumber of target cell IDs:")
        print(len(cell_ids))

        print("First five target cell IDs:")
        print(cell_ids[:5])

        print("\nFirst five RNA measurements for the first cell:")

        for column_number in range(5):
            print(
                "Cell:", cell_ids[0],
                "| Gene:", gene_names[column_number],
                "| Value:", values[0, column_number]
            )
def compare_multiome_input_target_ids():
    """
    Check that ATAC and RNA rows refer to the same cells.
    """

    input_path = DATA_DIR / "train_multi_inputs.h5"
    target_path = DATA_DIR / "train_multi_targets.h5"

    with h5py.File(input_path, "r") as input_file, \
         h5py.File(target_path, "r") as target_file:

        input_data = input_file["train_multi_inputs"]
        target_data = target_file["train_multi_targets"]

        input_cell_ids = decode_labels(
            input_data["axis1"][:]
        )

        target_cell_ids = decode_labels(
            target_data["axis1"][:]
        )

        print("\n\nMULTIOME INPUT-TARGET CHECK")

        print("Number of input cells:")
        print(len(input_cell_ids))

        print("Number of target cells:")
        print(len(target_cell_ids))

        print("\nSame number of cells:")
        print(len(input_cell_ids) == len(target_cell_ids))

        print("\nSame cell IDs in the same order:")
        print(input_cell_ids == target_cell_ids)

        print("\nFirst five row comparisons:")

        for row_number in range(5):
            print(
                "Row:", row_number,
                "| Input:", input_cell_ids[row_number],
                "| Target:", target_cell_ids[row_number],
                "| Match:",
                input_cell_ids[row_number]
                == target_cell_ids[row_number]
            )
def inspect_multiome_split(
    file_name,
    group_name,
    split_name
):
    """
    Show donors, days, and cell types in a Multiome split.
    """

    metadata_path = DATA_DIR / "metadata.csv"
    file_path = DATA_DIR / file_name

    metadata = pd.read_csv(metadata_path)
    metadata["cell_id"] = metadata["cell_id"].astype(str)
    metadata = metadata.set_index("cell_id")

    with h5py.File(file_path, "r") as file:

        data = file[group_name]

        cell_ids = decode_labels(
            data["axis1"][:]
        )

    split_metadata = metadata.loc[cell_ids]

    print("\n\nMULTIOME", split_name, "SPLIT")

    print("\nNumber of cells:")
    print(len(split_metadata))

    print("\nDonors:")
    print(sorted(split_metadata["donor"].unique()))

    print("\nDays:")
    print(sorted(split_metadata["day"].unique()))

    print("\nCells per donor and day:")
    print(
        pd.crosstab(
            split_metadata["donor"],
            split_metadata["day"]
        )
    )

    print("\nCell types:")
    print(
        split_metadata["cell_type"].value_counts()
    )

def save_pca_scatter_plot(
    train_components,
    validation_components,
    title,
    output_path
):
    """
    Plot the first two PCA components.

    Blue:
        Training cells

    Orange:
        Held-out validation cells
    """

    plt.figure(figsize=(7, 5))

    plt.scatter(
        train_components[:, 0],
        train_components[:, 1],
        s=12,
        alpha=0.45,
        label="Training cells",
        color="#2E6F9E"
    )

    plt.scatter(
        validation_components[:, 0],
        validation_components[:, 1],
        s=18,
        alpha=0.70,
        label="Held-out donor 31800",
        color="#C66A2B"
    )

    plt.xlabel("Principal component 1")
    plt.ylabel("Principal component 2")
    plt.title(title)

    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=220
    )

    plt.close()


if __name__ == "__main__":
    inspect_multiome_input()
    inspect_multiome_target()
    compare_multiome_input_target_ids()

    inspect_multiome_split(
        "train_multi_inputs.h5",
        "train_multi_inputs",
        "TRAIN"
    )

    inspect_multiome_split(
        "test_multi_inputs.h5",
        "test_multi_inputs",
        "TEST"
    )
from pathlib import Path

import h5py
import hdf5plugin  # Lets Python read the compressed Kaggle file.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Kaggle downloaded the data into this folder.
DATA_DIR = Path(
    "/home/ibab/.cache/kagglehub/competitions/open-problems-multimodal"
)


# The names inside the HDF5 file are stored as bytes.
# This function changes them into normal text.
def decode_labels(values):
    labels = []

    for value in values:
        labels.append(
            value.decode("utf-8").rstrip("\x00")
        )

    return labels


# This is the first file we are learning to read.
def inspect_cite_input():

    file_path = DATA_DIR / "train_cite_inputs.h5"

    # Open the file in read-only mode.
    with h5py.File(file_path, "r") as file:

        # Select the CITE-seq group inside the HDF5 file.
        data = file["train_cite_inputs"]

        # This is the actual RNA matrix.
        values = data["block0_values"]

        # axis0 contains the column names: RNA features.
        feature_names = decode_labels(
            data["axis0"][:]
        )

        # axis1 contains the row names: cell IDs.
        cell_ids = decode_labels(
            data["axis1"][:]
        )

        print("File:", file_path.name)
        print("Matrix shape:", values.shape)
        print("Data type:", values.dtype)

        print("\nNumber of feature names:")
        print(len(feature_names))

        print("First five feature names:")
        print(feature_names[:5])

        print("\nNumber of cell IDs:")
        print(len(cell_ids))

        print("First five cell IDs:")
        print(cell_ids[:5])

        # Show the first five columns for the first cell.
        print("\nFirst five measurements for the first cell:")

        for column_number in range(5):
            print(
                "Cell:", cell_ids[0],
                "| Feature:", feature_names[column_number],
                "| Value:", values[0, column_number]
            )

        # Read all RNA values for the first cell.
        first_cell_values = values[0, :]

        # Find the columns where this cell has a nonzero value.
        nonzero_columns = np.flatnonzero(
            first_cell_values
        )

        print("\nNumber of nonzero features in the first cell:")
        print(len(nonzero_columns))

        print("\nFirst ten nonzero measurements:")

        for column_number in nonzero_columns[:10]:
            print(
                "Cell:", cell_ids[0],
                "| Column:", column_number,
                "| Feature:", feature_names[column_number],
                "| Value:", first_cell_values[column_number]
            )
# Read the Protein Target File
def inspect_cite_target():

    file_path = DATA_DIR / "train_cite_targets.h5"

    with h5py.File(file_path, "r") as file:

        # Select the protein target group.
        data = file["train_cite_targets"]

        # This is the protein matrix.
        values = data["block0_values"]

        # axis0 contains protein names.
        protein_names = decode_labels(
            data["axis0"][:]
        )

        # axis1 contains cell IDs.
        cell_ids = decode_labels(
            data["axis1"][:]
        )

        print("\n\nTARGET FILE:", file_path.name)
        print("Target matrix shape:", values.shape)
        print("Target data type:", values.dtype)

        print("\nNumber of protein names:")
        print(len(protein_names))

        print("First five protein names:")
        print(protein_names[:5])

        print("\nNumber of target cell IDs:")
        print(len(cell_ids))

        print("First five target cell IDs:")
        print(cell_ids[:5])

        print("\nFirst five protein measurements for the first cell:")

        for column_number in range(5):
            print(
                "Cell:", cell_ids[0],
                "| Protein:", protein_names[column_number],
                "| Value:", values[0, column_number]
            )
#Compare all cell IDs
def compare_input_target_ids():

    input_path = DATA_DIR / "train_cite_inputs.h5"
    target_path = DATA_DIR / "train_cite_targets.h5"

    with h5py.File(input_path, "r") as input_file, \
         h5py.File(target_path, "r") as target_file:

        input_data = input_file["train_cite_inputs"]
        target_data = target_file["train_cite_targets"]

        input_cell_ids = decode_labels(
            input_data["axis1"][:]
        )

        target_cell_ids = decode_labels(
            target_data["axis1"][:]
        )

        print("\n\nINPUT-TARGET CELL CHECK")

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
#understand the cell metadata
def inspect_metadata_for_first_cells():

    metadata_path = DATA_DIR / "metadata.csv"

    # Read the table containing donor, day, and cell-type information.
    metadata = pd.read_csv(metadata_path)

    # Make sure cell IDs have the same text format as the HDF5 IDs.
    metadata["cell_id"] = metadata["cell_id"].astype(str)

    input_path = DATA_DIR / "train_cite_inputs.h5"

    with h5py.File(input_path, "r") as file:

        data = file["train_cite_inputs"]

        first_cell_ids = decode_labels(
            data["axis1"][:5]
        )

    metadata_by_cell = metadata.set_index("cell_id")

    first_cells = metadata_by_cell.loc[first_cell_ids]

    print("\n\nMETADATA FOR THE FIRST FIVE CELLS")
    print(first_cells)

# lets inspect the training split
def inspect_training_split():

    metadata_path = DATA_DIR / "metadata.csv"
    input_path = DATA_DIR / "train_cite_inputs.h5"

    metadata = pd.read_csv(metadata_path)
    metadata["cell_id"] = metadata["cell_id"].astype(str)
    metadata = metadata.set_index("cell_id")

    # Read only the training cell IDs from the HDF5 file.
    with h5py.File(input_path, "r") as file:

        data = file["train_cite_inputs"]

        training_cell_ids = decode_labels(
            data["axis1"][:]
        )

    # Keep metadata only for cells in train_cite_inputs.h5.
    training_metadata = metadata.loc[training_cell_ids]

    print("\n\nCITE-SEQ TRAINING SPLIT")

    print("\nNumber of training cells:")
    print(len(training_metadata))

    print("\nTraining donors:")
    print(sorted(training_metadata["donor"].unique()))

    print("\nTraining days:")
    print(sorted(training_metadata["day"].unique()))

    print("\nCells per donor and day:")
    print(
        pd.crosstab(
            training_metadata["donor"],
            training_metadata["day"]
        )
    )
def inspect_training_split():

    metadata_path = DATA_DIR / "metadata.csv"
    input_path = DATA_DIR / "train_cite_inputs.h5"

    metadata = pd.read_csv(metadata_path)
    metadata["cell_id"] = metadata["cell_id"].astype(str)
    metadata = metadata.set_index("cell_id")

    # Read only the training cell IDs from the HDF5 file.
    with h5py.File(input_path, "r") as file:

        data = file["train_cite_inputs"]

        training_cell_ids = decode_labels(
            data["axis1"][:]
        )

    # Keep metadata only for cells in train_cite_inputs.h5.
    training_metadata = metadata.loc[training_cell_ids]

    print("\n\nCITE-SEQ TRAINING SPLIT")

    print("\nNumber of training cells:")
    print(len(training_metadata))

    print("\nTraining donors:")
    print(sorted(training_metadata["donor"].unique()))

    print("\nTraining days:")
    print(sorted(training_metadata["day"].unique()))

    print("\nCells per donor and day:")
    print(
        pd.crosstab(
            training_metadata["donor"],
            training_metadata["day"]
        )
    )
# Inspect the cite seq test split
def inspect_test_split():

    metadata_path = DATA_DIR / "metadata.csv"
    test_path = DATA_DIR / "test_cite_inputs.h5"

    metadata = pd.read_csv(metadata_path)
    metadata["cell_id"] = metadata["cell_id"].astype(str)
    metadata = metadata.set_index("cell_id")

    # Read only the test cell IDs.
    with h5py.File(test_path, "r") as file:

        data = file["test_cite_inputs"]

        test_cell_ids = decode_labels(
            data["axis1"][:]
        )

    test_metadata = metadata.loc[test_cell_ids]

    print("\n\nCITE-SEQ TEST SPLIT")

    print("\nNumber of test cells:")
    print(len(test_metadata))

    print("\nTest donors:")
    print(sorted(test_metadata["donor"].unique()))

    print("\nTest days:")
    print(sorted(test_metadata["day"].unique()))

    print("\nCells per donor and day:")
    print(
        pd.crosstab(
            test_metadata["donor"],
            test_metadata["day"]
        )
    )

    print("\nTest cell types:")
    print(
        test_metadata["cell_type"].value_counts()
    )

# Start the program here.
if __name__ == "__main__":
    inspect_cite_input()
    inspect_cite_target()
    compare_input_target_ids()
    inspect_metadata_for_first_cells()
    inspect_training_split()
    inspect_test_split()



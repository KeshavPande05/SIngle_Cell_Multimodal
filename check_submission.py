import csv
import gc
import sys
from pathlib import Path

import h5py
import hdf5plugin
import numpy as np
import torch


PROJECT_DIR = Path(__file__).resolve().parent
SRC_DIR = PROJECT_DIR / "src"
DATA_DIR = Path(
    "/home/ibab/.cache/kagglehub/competitions/open-problems-multimodal"
)

OUTPUT_PATH = PROJECT_DIR / "submission.csv"
EVALUATION_PATH = DATA_DIR / "evaluation_ids.csv"

BATCH_CELLS = 64

# Make the existing experiment modules importable.
sys.path.insert(0, str(SRC_DIR))

from cite_deep_learning import (  # noqa: E402
    choose_cite_rows,
    decode_labels as decode_cite_labels,
    fit_cite_ridge_model,
    read_cite_rows,
    reduce_cite_inputs,
)
from multiome_baseline import (  # noqa: E402
    choose_multiome_rows,
    decode_labels as decode_multiome_labels,
    fit_mlp_model,
    read_multiome_rows,
    reduce_multiome_data,
)


# -----------------------------------------------------------------------------
# Model preparation
# -----------------------------------------------------------------------------


def train_cite_submission_model():
    """
    Train the selected CITE-seq Ridge model on the reproducible 1,000-cell sample.
    """

    print("\nPreparing CITE-seq model...")

    train_rows, validation_rows = choose_cite_rows(
        validation_donor=31800,
        max_train=1000,
        max_validation=300,
    )

    X_train, Y_train = read_cite_rows(train_rows)
    X_validation, Y_validation = read_cite_rows(validation_rows)

    (
        X_train_reduced,
        X_validation_reduced,
        selected_genes,
        rna_pca,
    ) = reduce_cite_inputs(
        X_train,
        X_validation,
        number_of_genes=5000,
        number_of_components=128,
    )

    ridge_model, input_scaler, _ = fit_cite_ridge_model(
        X_train_reduced,
        X_validation_reduced,
        Y_train,
        Y_validation,
    )

    del X_train
    del X_validation
    del Y_train
    del Y_validation
    del X_train_reduced
    del X_validation_reduced
    gc.collect()

    return {
        "model": ridge_model,
        "input_scaler": input_scaler,
        "rna_pca": rna_pca,
        "selected_genes": selected_genes,
    }


def train_multiome_submission_model():
    """
    Train the selected Multiome MLP on the reproducible 1,000-cell sample.
    """

    print("\nPreparing Multiome model...")

    train_rows, validation_rows = choose_multiome_rows(
        validation_donor=31800,
        max_train=1000,
        max_validation=300,
    )

    X_train, Y_train = read_multiome_rows(train_rows)
    X_validation, Y_validation = read_multiome_rows(validation_rows)

    (
        X_train_reduced,
        X_validation_reduced,
        Y_train_reduced,
        Y_validation_reduced,
        selected_atac_features,
        rna_pca,
    ) = reduce_multiome_data(
        X_train,
        Y_train,
        X_validation,
        Y_validation,
        number_of_atac_features=5000,
        number_of_rna_components=64,
    )

    (
        model,
        input_scaler,
        target_scaler,
        _,
    ) = fit_mlp_model(
        X_train_reduced,
        Y_train_reduced,
        X_validation_reduced,
        Y_validation_reduced,
        Y_train,
        Y_validation,
        rna_pca,
    )

    del X_train
    del X_validation
    del Y_train
    del Y_validation
    del X_train_reduced
    del X_validation_reduced
    del Y_train_reduced
    del Y_validation_reduced
    gc.collect()

    model.eval()

    return {
        "model": model,
        "input_scaler": input_scaler,
        "target_scaler": target_scaler,
        "rna_pca": rna_pca,
        "selected_atac_features": selected_atac_features,
    }


# -----------------------------------------------------------------------------
# HDF5 and feature helpers
# -----------------------------------------------------------------------------


def read_axis_labels(file_path, group_name, axis_name):
    with h5py.File(file_path, "r") as file:
        values = file[group_name][axis_name][:]

    return decode_cite_labels(values)


def read_rows_in_original_order(dataset, row_indices):
    """
    h5py expects fancy-index row positions to be sorted.
    This helper restores the requested order after reading.
    """

    row_indices = np.asarray(row_indices, dtype=np.int64)
    sort_order = np.argsort(row_indices)
    sorted_rows = row_indices[sort_order]

    values_sorted = np.asarray(
        dataset[sorted_rows, :],
        dtype=np.float32,
    )

    restore_order = np.argsort(sort_order)

    return values_sorted[restore_order]


def prepare_test_metadata():
    cite_input_path = DATA_DIR / "test_cite_inputs.h5"
    multiome_input_path = DATA_DIR / "test_multi_inputs.h5"

    cite_target_path = DATA_DIR / "train_cite_targets.h5"
    multiome_target_path = DATA_DIR / "train_multi_targets.h5"

    cite_test_cells = read_axis_labels(
        cite_input_path,
        "test_cite_inputs",
        "axis1",
    )

    multiome_test_cells = read_axis_labels(
        multiome_input_path,
        "test_multi_inputs",
        "axis1",
    )

    cite_proteins = read_axis_labels(
        cite_target_path,
        "train_cite_targets",
        "axis0",
    )

    multiome_genes = read_axis_labels(
        multiome_target_path,
        "train_multi_targets",
        "axis0",
    )

    return {
        "cite_test_row": {
            cell_id: index
            for index, cell_id in enumerate(cite_test_cells)
        },
        "multiome_test_row": {
            cell_id: index
            for index, cell_id in enumerate(multiome_test_cells)
        },
        "cite_feature": {
            feature: index
            for index, feature in enumerate(cite_proteins)
        },
        "multiome_feature": {
            feature: index
            for index, feature in enumerate(multiome_genes)
        },
    }


def classify_feature(feature, metadata):
    if feature in metadata["cite_feature"]:
        return "cite", metadata["cite_feature"][feature]

    if feature in metadata["multiome_feature"]:
        return "multiome", metadata["multiome_feature"][feature]

    raise KeyError(
        f"Feature {feature!r} was not found in either target feature list."
    )


# -----------------------------------------------------------------------------
# Batched test prediction
# -----------------------------------------------------------------------------


def predict_cite_cells(cell_ids, model_info, metadata, input_dataset):
    rows = [metadata["cite_test_row"][cell_id] for cell_id in cell_ids]

    input_values = read_rows_in_original_order(
        input_dataset,
        rows,
    )

    selected = input_values[:, model_info["selected_genes"]]
    reduced = model_info["rna_pca"].transform(selected)
    scaled = model_info["input_scaler"].transform(reduced)

    predictions = model_info["model"].predict(scaled)

    return predictions


def predict_multiome_cells(cell_ids, model_info, metadata, input_dataset):
    rows = [metadata["multiome_test_row"][cell_id] for cell_id in cell_ids]

    input_values = read_rows_in_original_order(
        input_dataset,
        rows,
    )

    selected = input_values[:, model_info["selected_atac_features"]]
    scaled = model_info["input_scaler"].transform(selected)

    input_tensor = torch.tensor(
        scaled.astype(np.float32)
    )

    with torch.no_grad():
        predicted_scaled = model_info["model"](
            input_tensor
        ).numpy()

    predicted_latent = model_info["target_scaler"].inverse_transform(
        predicted_scaled
    )

    predictions = model_info["rna_pca"].inverse_transform(
        predicted_latent
    )

    return predictions


def write_group_predictions(
    groups,
    writer,
    metadata,
    cite_model,
    multiome_model,
    cite_input_dataset,
    multiome_input_dataset,
):
    """
    Predict one batch of contiguous cell groups and write rows in order.
    """

    if not groups:
        return

    modality = groups[0]["modality"]
    cell_ids = [group["cell_id"] for group in groups]

    if modality == "cite":
        predictions = predict_cite_cells(
            cell_ids,
            cite_model,
            metadata,
            cite_input_dataset,
        )
    else:
        predictions = predict_multiome_cells(
            cell_ids,
            multiome_model,
            metadata,
            multiome_input_dataset,
        )

    for group_index, group in enumerate(groups):
        row_predictions = predictions[group_index]

        for row_id, feature_index in zip(
            group["row_ids"],
            group["feature_indices"],
        ):
            value = float(row_predictions[feature_index])
            writer.writerow([row_id, f"{value:.8g}"])


def create_submission():
    metadata = prepare_test_metadata()

    cite_model = train_cite_submission_model()
    multiome_model = train_multiome_submission_model()

    cite_input_path = DATA_DIR / "test_cite_inputs.h5"
    multiome_input_path = DATA_DIR / "test_multi_inputs.h5"

    print("\nWriting submission:", OUTPUT_PATH)
    print("Evaluation file:", EVALUATION_PATH)

    groups = []
    current_group = None
    total_rows = 0

    with h5py.File(cite_input_path, "r") as cite_file, \
         h5py.File(multiome_input_path, "r") as multiome_file, \
         EVALUATION_PATH.open("r", newline="") as evaluation_file, \
         OUTPUT_PATH.open("w", newline="") as submission_file:

        cite_input_dataset = cite_file[
            "test_cite_inputs"
        ]["block0_values"]

        multiome_input_dataset = multiome_file[
            "test_multi_inputs"
        ]["block0_values"]

        evaluation_reader = csv.DictReader(evaluation_file)
        writer = csv.writer(submission_file)
        writer.writerow(["row_id", "target"])

        for record in evaluation_reader:
            row_id = record["row_id"]
            cell_id = record["cell_id"]
            feature = record["gene_id"]

            modality, feature_index = classify_feature(
                feature,
                metadata,
            )

            group_key = (cell_id, modality)

            if current_group is None:
                current_group = {
                    "cell_id": cell_id,
                    "modality": modality,
                    "row_ids": [],
                    "feature_indices": [],
                }

            current_key = (
                current_group["cell_id"],
                current_group["modality"],
            )

            if group_key != current_key:
                groups.append(current_group)
                current_group = {
                    "cell_id": cell_id,
                    "modality": modality,
                    "row_ids": [],
                    "feature_indices": [],
                }

                # Keep each prediction batch within one modality.
                if (
                    len(groups) >= BATCH_CELLS
                    or groups[0]["modality"] != modality
                ):
                    write_group_predictions(
                        groups,
                        writer,
                        metadata,
                        cite_model,
                        multiome_model,
                        cite_input_dataset,
                        multiome_input_dataset,
                    )
                    total_rows += sum(
                        len(group["row_ids"])
                        for group in groups
                    )
                    groups = []

            current_group["row_ids"].append(row_id)
            current_group["feature_indices"].append(feature_index)

        if current_group is not None:
            groups.append(current_group)

        if groups:
            write_group_predictions(
                groups,
                writer,
                metadata,
                cite_model,
                multiome_model,
                cite_input_dataset,
                multiome_input_dataset,
            )
            total_rows += sum(
                len(group["row_ids"])
                for group in groups
            )

    print("Rows written:", total_rows)
    print("Submission created:", OUTPUT_PATH)


if __name__ == "__main__":
    create_submission()

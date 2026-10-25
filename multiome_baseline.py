from pathlib import Path

import h5py
import hdf5plugin
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


DATA_DIR = Path(
    "/home/ibab/.cache/kagglehub/competitions/open-problems-multimodal"
)

PROJECT_DIR = Path(__file__).resolve().parent
PLOT_DIR = PROJECT_DIR / "multiome_plots"
PLOT_DIR.mkdir(exist_ok=True)


# -----------------------------------------------------------------------------
# Data reading
# -----------------------------------------------------------------------------


def decode_labels(values):
    """Convert byte labels stored in HDF5 into normal strings."""

    labels = []

    for value in values:
        if isinstance(value, bytes):
            value = value.decode("utf-8")

        labels.append(str(value).rstrip("\x00"))

    return labels


def choose_multiome_rows(
    validation_donor=31800,
    max_train=1000,
    max_validation=300
):
    """Select training and validation rows using donor separation."""

    input_path = DATA_DIR / "train_multi_inputs.h5"
    metadata_path = DATA_DIR / "metadata.csv"

    metadata = pd.read_csv(metadata_path)
    metadata["cell_id"] = metadata["cell_id"].astype(str)
    metadata = metadata.set_index("cell_id")

    with h5py.File(input_path, "r") as file:
        data = file["train_multi_inputs"]
        matrix_shape = data["block0_values"].shape
        all_cell_ids = decode_labels(data["axis1"][:])

    # Match metadata to the exact HDF5 row order.
    cell_metadata = metadata.loc[all_cell_ids]
    donor_values = cell_metadata["donor"].to_numpy()

    training_pool = np.flatnonzero(
        donor_values != validation_donor
    )

    validation_pool = np.flatnonzero(
        donor_values == validation_donor
    )

    random = np.random.default_rng(42)

    train_rows = random.choice(
        training_pool,
        size=min(max_train, len(training_pool)),
        replace=False
    )

    validation_rows = random.choice(
        validation_pool,
        size=min(max_validation, len(validation_pool)),
        replace=False
    )

    train_rows.sort()
    validation_rows.sort()

    print("Multiome input matrix shape:")
    print(matrix_shape)

    print("\nValidation donor:")
    print(validation_donor)

    print("\nTraining rows selected:")
    print(len(train_rows))

    print("Validation rows selected:")
    print(len(validation_rows))

    print("\nTraining donors:")
    print(
        sorted(
            int(value)
            for value in cell_metadata.iloc[train_rows]["donor"].unique()
        )
    )

    print("Validation donors:")
    print(
        sorted(
            int(value)
            for value in cell_metadata.iloc[validation_rows]["donor"].unique()
        )
    )

    print("\nFirst three training cells:")
    print(cell_metadata.iloc[train_rows[:3]])

    print("\nFirst three validation cells:")
    print(cell_metadata.iloc[validation_rows[:3]])

    return train_rows, validation_rows


def read_multiome_rows(row_indices):
    """Read aligned ATAC input rows and RNA target rows."""

    input_path = DATA_DIR / "train_multi_inputs.h5"
    target_path = DATA_DIR / "train_multi_targets.h5"

    with h5py.File(input_path, "r") as input_file, \
         h5py.File(target_path, "r") as target_file:

        input_values = input_file[
            "train_multi_inputs"
        ]["block0_values"]

        target_values = target_file[
            "train_multi_targets"
        ]["block0_values"]

        X = np.asarray(
            input_values[row_indices, :],
            dtype=np.float32
        )

        Y = np.asarray(
            target_values[row_indices, :],
            dtype=np.float32
        )

    return X, Y


# -----------------------------------------------------------------------------
# Reduction
# -----------------------------------------------------------------------------


def reduce_multiome_data(
    X_train,
    Y_train,
    X_validation,
    Y_validation,
    number_of_atac_features=5000,
    number_of_rna_components=64
):
    """
    Select variable ATAC regions and compress RNA targets with PCA.

    Important:
        PCA is applied to the RNA target in this baseline.
        The ATAC input is reduced by variance selection only.
    """

    atac_variance = np.var(
        X_train,
        axis=0
    )

    selected_atac_features = np.argsort(
        atac_variance
    )[-number_of_atac_features:]

    selected_atac_features.sort()

    X_train_reduced = X_train[
        :,
        selected_atac_features
    ]

    X_validation_reduced = X_validation[
        :,
        selected_atac_features
    ]

    rna_pca = PCA(
        n_components=number_of_rna_components,
        svd_solver="randomized",
        random_state=42
    )

    Y_train_reduced = rna_pca.fit_transform(
        Y_train
    )

    Y_validation_reduced = rna_pca.transform(
        Y_validation
    )

    return (
        X_train_reduced,
        X_validation_reduced,
        Y_train_reduced,
        Y_validation_reduced,
        selected_atac_features,
        rna_pca
    )


# -----------------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------------


def save_pca_scatter_plot(
    train_components,
    validation_components,
    title,
    output_path
):
    """Plot PC1 versus PC2 for training and validation cells."""

    plt.figure(figsize=(7, 5))

    plt.scatter(
        train_components[:, 0],
        train_components[:, 1],
        s=12,
        alpha=0.45,
        color="#2E6F9E",
        label="Training cells"
    )

    plt.scatter(
        validation_components[:, 0],
        validation_components[:, 1],
        s=18,
        alpha=0.70,
        color="#C66A2B",
        label="Held-out donor 31800"
    )

    plt.xlabel("Principal component 1")
    plt.ylabel("Principal component 2")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def save_pca_variance_plot(
    pca,
    title,
    output_path,
    number_to_show=20
):
    """Plot individual and cumulative PCA variance."""

    number_to_show = min(
        number_to_show,
        len(pca.explained_variance_ratio_)
    )

    component_numbers = np.arange(
        1,
        number_to_show + 1
    )

    explained_variance = (
        pca.explained_variance_ratio_[:number_to_show]
        * 100
    )

    cumulative_variance = np.cumsum(
        explained_variance
    )

    figure, axis = plt.subplots(
        figsize=(7, 5)
    )

    axis.bar(
        component_numbers,
        explained_variance,
        color="#5B238B",
        alpha=0.75,
        label="Individual variance"
    )

    axis.set_xlabel("Principal component")
    axis.set_ylabel("Variance explained (%)")
    axis.set_title(title)
    axis.grid(axis="y", alpha=0.25)

    second_axis = axis.twinx()

    second_axis.plot(
        component_numbers,
        cumulative_variance,
        color="#C66A2B",
        marker="o",
        label="Cumulative variance"
    )

    second_axis.set_ylabel(
        "Cumulative variance (%)"
    )

    figure.tight_layout()
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def create_multiome_pca_plots(
    X_train_reduced,
    X_validation_reduced,
    Y_train_reduced,
    Y_validation_reduced,
    rna_pca
):
    """
    Create the two Multiome PCA views.

    1. RNA target PCA: the PCA actually used by the model.
    2. ATAC visualisation PCA: an extra two-dimensional plot only
       for visualisation; it is not used as the model input.
    """

    save_pca_scatter_plot(
        Y_train_reduced,
        Y_validation_reduced,
        "Multiome target RNA PCA: training versus held-out donor",
        PLOT_DIR / "multiome_target_rna_pca_scatter.png"
    )

    save_pca_variance_plot(
        rna_pca,
        "Multiome target RNA PCA explained variance",
        PLOT_DIR / "multiome_target_rna_pca_variance.png"
    )

    # This PCA is only for plotting selected ATAC regions in two dimensions.
    # The model still uses all 5,000 selected ATAC regions.
    atac_visual_pca = PCA(
        n_components=2,
        svd_solver="randomized",
        random_state=42
    )

    X_train_atac_visual = atac_visual_pca.fit_transform(
        X_train_reduced
    )

    X_validation_atac_visual = atac_visual_pca.transform(
        X_validation_reduced
    )

    save_pca_scatter_plot(
        X_train_atac_visual,
        X_validation_atac_visual,
        "Multiome ATAC visualisation PCA",
        PLOT_DIR / "multiome_atac_visual_pca_scatter.png"
    )

    save_pca_variance_plot(
        atac_visual_pca,
        "Multiome ATAC visualisation PCA variance",
        PLOT_DIR / "multiome_atac_visual_pca_variance.png",
        number_to_show=2
    )


# -----------------------------------------------------------------------------
# Metrics and models
# -----------------------------------------------------------------------------


def calculate_rmse(actual, predicted):
    """Calculate overall root mean squared error."""

    return np.sqrt(
        np.mean(
            (actual - predicted) ** 2
        )
    )


def calculate_mean_gene_pearson(actual, predicted):
    """Calculate the earlier feature-wise diagnostic Pearson value."""

    actual_centered = (
        actual - actual.mean(axis=0)
    )

    predicted_centered = (
        predicted - predicted.mean(axis=0)
    )

    numerator = np.sum(
        actual_centered * predicted_centered,
        axis=0
    )

    actual_size = np.sqrt(
        np.sum(actual_centered ** 2, axis=0)
    )

    predicted_size = np.sqrt(
        np.sum(predicted_centered ** 2, axis=0)
    )

    denominator = actual_size * predicted_size
    valid_genes = denominator > 0

    if not np.any(valid_genes):
        return np.nan

    correlations = (
        numerator[valid_genes]
        / denominator[valid_genes]
    )

    return correlations.mean()


class MultiomeMLP(nn.Module):
    """Small neural network for predicting RNA PCA components."""

    def __init__(
        self,
        number_of_input_features,
        number_of_target_components
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(
                number_of_input_features,
                128
            ),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(
                128,
                number_of_target_components
            )
        )

    def forward(self, inputs):
        return self.network(inputs)


def fit_mlp_model(
    X_train_reduced,
    Y_train_reduced,
    X_validation_reduced,
    Y_validation_reduced,
    Y_train_raw,
    Y_validation_raw,
    rna_pca
):
    """Train and evaluate the Multiome MLP."""

    torch.manual_seed(42)

    input_scaler = StandardScaler()

    X_train_scaled = input_scaler.fit_transform(
        X_train_reduced
    )

    X_validation_scaled = input_scaler.transform(
        X_validation_reduced
    )

    target_scaler = StandardScaler()

    Y_train_scaled = target_scaler.fit_transform(
        Y_train_reduced
    )

    X_train_tensor = torch.tensor(
        X_train_scaled.astype(np.float32)
    )

    Y_train_tensor = torch.tensor(
        Y_train_scaled.astype(np.float32)
    )

    X_validation_tensor = torch.tensor(
        X_validation_scaled.astype(np.float32)
    )

    train_dataset = TensorDataset(
        X_train_tensor,
        Y_train_tensor
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True
    )

    model = MultiomeMLP(
        number_of_input_features=X_train_reduced.shape[1],
        number_of_target_components=Y_train_reduced.shape[1]
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
        weight_decay=0.0001
    )

    loss_function = nn.MSELoss()
    loss_history = []

    for epoch in range(1, 101):
        model.train()
        total_loss = 0.0

        for batch_inputs, batch_targets in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_inputs)
            loss = loss_function(predictions, batch_targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_inputs.shape[0]

        average_loss = total_loss / len(train_dataset)
        loss_history.append(
            average_loss
        )
        plt.figure(
            figsize=(7, 5)
        )

        plt.plot(
            range(1, len(loss_history) + 1),
            loss_history,
            color="#C66A2B",
            linewidth=2
        )

        plt.xlabel("Training epoch")
        plt.ylabel("Training MSE")
        plt.title("Multiome MLP training loss")

        plt.grid(
            alpha=0.25
        )

        plt.tight_layout()

        plt.savefig(
            PLOT_DIR / "multiome_mlp_training_loss.png",
            dpi=220
        )

        plt.close()

        if epoch == 1 or epoch % 20 == 0:
            print(
                "MLP epoch:",
                epoch,
                "training loss:",
                average_loss
            )

    model.eval()

    with torch.no_grad():
        predicted_validation_scaled = model(
            X_validation_tensor
        ).numpy()

    predicted_validation_latent = target_scaler.inverse_transform(
        predicted_validation_scaled
    )

    predicted_validation_rna = rna_pca.inverse_transform(
        predicted_validation_latent
    )

    model_rmse = calculate_rmse(
        Y_validation_raw,
        predicted_validation_rna
    )

    model_pearson = calculate_mean_gene_pearson(
        Y_validation_raw,
        predicted_validation_rna
    )

    print("\nMULTIOME MLP")
    print("Model RNA RMSE:", model_rmse)
    print("Model mean gene Pearson correlation:", model_pearson)

    return (
        model,
        input_scaler,
        target_scaler,
        predicted_validation_rna
    )


def fit_ridge_model(
    X_train_reduced,
    Y_train_reduced,
    X_validation_reduced,
    Y_validation_reduced,
    Y_train_raw,
    Y_validation_raw,
    rna_pca
):
    """Train Ridge on 5,000 ATAC regions and 64 RNA PCA targets."""

    input_scaler = StandardScaler()

    X_train_scaled = input_scaler.fit_transform(
        X_train_reduced
    )

    X_validation_scaled = input_scaler.transform(
        X_validation_reduced
    )

    ridge_model = Ridge(
        alpha=10.0,
        solver="lsqr"
    )

    ridge_model.fit(
        X_train_scaled,
        Y_train_reduced
    )

    predicted_validation_latent = ridge_model.predict(
        X_validation_scaled
    )

    predicted_validation_rna = rna_pca.inverse_transform(
        predicted_validation_latent
    )

    mean_rna_prediction = np.tile(
        Y_train_raw.mean(axis=0),
        (Y_validation_raw.shape[0], 1)
    )

    model_rmse = calculate_rmse(
        Y_validation_raw,
        predicted_validation_rna
    )

    baseline_rmse = calculate_rmse(
        Y_validation_raw,
        mean_rna_prediction
    )

    model_pearson = calculate_mean_gene_pearson(
        Y_validation_raw,
        predicted_validation_rna
    )

    print("\nRIDGE BASELINE")
    print("Model RNA RMSE:", model_rmse)
    print("Mean-expression baseline RMSE:", baseline_rmse)
    print("Model mean gene Pearson correlation:", model_pearson)

    return (
        ridge_model,
        input_scaler,
        predicted_validation_rna
    )


# -----------------------------------------------------------------------------
# Main experiment
# -----------------------------------------------------------------------------


if __name__ == "__main__":

    train_rows, validation_rows = choose_multiome_rows()

    X_train, Y_train = read_multiome_rows(train_rows)
    X_validation, Y_validation = read_multiome_rows(validation_rows)

    (
        X_train_reduced,
        X_validation_reduced,
        Y_train_reduced,
        Y_validation_reduced,
        selected_atac_features,
        rna_pca
    ) = reduce_multiome_data(
        X_train,
        Y_train,
        X_validation,
        Y_validation
    )

    print("\nREDUCED DATA SHAPES")
    print("X_train_reduced:", X_train_reduced.shape)
    print("X_validation_reduced:", X_validation_reduced.shape)
    print("Y_train_reduced:", Y_train_reduced.shape)
    print("Y_validation_reduced:", Y_validation_reduced.shape)

    print("\nNumber of selected ATAC features:", len(selected_atac_features))
    print(
        "RNA variance retained by PCA:",
        rna_pca.explained_variance_ratio_.sum()
    )

    print("\nREAD DATA SHAPES")
    print("X_train:", X_train.shape)
    print("Y_train:", Y_train.shape)
    print("X_validation:", X_validation.shape)
    print("Y_validation:", Y_validation.shape)

    print(
        "\nNonzero ATAC regions in first training cell:",
        np.count_nonzero(X_train[0])
    )

    print(
        "Nonzero RNA genes in first training cell:",
        np.count_nonzero(Y_train[0])
    )

    # These are the actual Multiome PCA plots.
    create_multiome_pca_plots(
        X_train_reduced,
        X_validation_reduced,
        Y_train_reduced,
        Y_validation_reduced,
        rna_pca
    )

    fit_ridge_model(
        X_train_reduced,
        Y_train_reduced,
        X_validation_reduced,
        Y_validation_reduced,
        Y_train,
        Y_validation,
        rna_pca
    )

    fit_mlp_model(
        X_train_reduced,
        Y_train_reduced,
        X_validation_reduced,
        Y_validation_reduced,
        Y_train,
        Y_validation,
        rna_pca
    )

    print("\nPCA plots saved in:")
    print(PLOT_DIR)

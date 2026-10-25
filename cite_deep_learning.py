from pathlib import Path

import h5py
import hdf5plugin
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


DATA_DIR = Path(
    "/home/ibab/.cache/kagglehub/competitions/open-problems-multimodal"
)


def decode_labels(values):
    """
    Convert labels stored as HDF5 bytes into text.
    """

    labels = []

    for value in values:

        if isinstance(value, bytes):
            value = value.decode("utf-8")

        labels.append(
            str(value).rstrip("\x00")
        )

    return labels


def choose_cite_rows(
    validation_donor=31800,
    max_train=1000,
    max_validation=300
):
    """
    Select CITE-seq training and validation rows.

    The validation donor is completely excluded from training.
    """

    input_path = DATA_DIR / "train_cite_inputs.h5"
    metadata_path = DATA_DIR / "metadata.csv"

    metadata = pd.read_csv(metadata_path)
    metadata["cell_id"] = metadata["cell_id"].astype(str)
    metadata = metadata.set_index("cell_id")

    with h5py.File(input_path, "r") as file:

        data = file["train_cite_inputs"]

        matrix_shape = data["block0_values"].shape
        all_cell_ids = decode_labels(
            data["axis1"][:]
        )

    # Match metadata rows to the HDF5 row order.
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

    print("CITE input matrix shape:")
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
            for value in cell_metadata.iloc[
                validation_rows
            ]["donor"].unique()
        )
    )

    print("\nFirst three training cells:")
    print(
        cell_metadata.iloc[train_rows[:3]]
    )

    print("\nFirst three validation cells:")
    print(
        cell_metadata.iloc[validation_rows[:3]]
    )

    return train_rows, validation_rows


def read_cite_rows(row_indices):
    """
    Read matching RNA inputs and protein targets.
    """

    input_path = DATA_DIR / "train_cite_inputs.h5"
    target_path = DATA_DIR / "train_cite_targets.h5"

    with h5py.File(input_path, "r") as input_file, \
         h5py.File(target_path, "r") as target_file:

        input_values = input_file[
            "train_cite_inputs"
        ]["block0_values"]

        target_values = target_file[
            "train_cite_targets"
        ]["block0_values"]

        # The CITE input and target rows are aligned.
        X = np.asarray(
            input_values[row_indices, :],
            dtype=np.float32
        )

        Y = np.asarray(
            target_values[row_indices, :],
            dtype=np.float32
        )

    return X, Y
def reduce_cite_inputs(
    X_train,
    X_validation,
    number_of_genes=5000,
    number_of_components=128
):
    """
    Reduce the RNA input matrix.

    First, select the most variable genes.
    Then, use PCA to create a smaller representation.
    """

    # Calculate gene variance using training cells only.
    gene_variance = np.var(
        X_train,
        axis=0
    )

    # Select the most variable RNA genes.
    selected_genes = np.argsort(
        gene_variance
    )[-number_of_genes:]

    selected_genes.sort()

    X_train_selected = X_train[
        :,
        selected_genes
    ]

    X_validation_selected = X_validation[
        :,
        selected_genes
    ]

    # Fit PCA only on training RNA data.
    rna_pca = PCA(
        n_components=number_of_components,
        svd_solver="randomized",
        random_state=42
    )

    X_train_reduced = rna_pca.fit_transform(
        X_train_selected
    )

    # Apply the training PCA to validation cells.
    X_validation_reduced = rna_pca.transform(
        X_validation_selected
    )

    return (
        X_train_reduced,
        X_validation_reduced,
        selected_genes,
        rna_pca
    )
def calculate_rmse(actual, predicted):
    """
    Calculate overall root mean squared error.
    """

    return np.sqrt(
        np.mean(
            (actual - predicted) ** 2
        )
    )


def calculate_mean_feature_pearson(actual, predicted):
    """
    Calculate mean Pearson correlation across proteins.

    Each column represents one protein.
    """

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

    valid_proteins = denominator > 0

    if not np.any(valid_proteins):
        return np.nan

    correlations = (
        numerator[valid_proteins]
        / denominator[valid_proteins]
    )

    return correlations.mean()

def fit_cite_ridge_model(
    X_train_reduced,
    X_validation_reduced,
    Y_train,
    Y_validation
):
    """
    Predict 140 proteins using Ridge regression.
    """

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
        Y_train
    )

    predicted_proteins = ridge_model.predict(
        X_validation_scaled
    )

    mean_protein_prediction = np.tile(
        Y_train.mean(axis=0),
        (Y_validation.shape[0], 1)
    )

    model_rmse = calculate_rmse(
        Y_validation,
        predicted_proteins
    )

    baseline_rmse = calculate_rmse(
        Y_validation,
        mean_protein_prediction
    )

    model_pearson = calculate_mean_feature_pearson(
        Y_validation,
        predicted_proteins
    )

    print("\nCITE RIDGE BASELINE")

    print(
        "Model protein RMSE:",
        model_rmse
    )

    print(
        "Mean-protein baseline RMSE:",
        baseline_rmse
    )

    print(
        "Model mean protein Pearson correlation:",
        model_pearson
    )

    return (
        ridge_model,
        input_scaler,
        predicted_proteins
    )
class CiteMLP(nn.Module):
    """
    Small neural network for predicting 140 proteins
    from 128 RNA PCA components.
    """

    def __init__(
        self,
        number_of_input_components,
        number_of_proteins
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(
                number_of_input_components,
                64
            ),
            nn.ReLU(),

            nn.Dropout(
                p=0.1
            ),

            nn.Linear(
                64,
                number_of_proteins
            )
        )

    def forward(self, inputs):
        return self.network(inputs)
def fit_cite_mlp_model(
    X_train_reduced,
    X_validation_reduced,
    Y_train,
    Y_validation
):
    """
    Train and evaluate the CITE-seq MLP.
    """

    torch.manual_seed(42)

    input_scaler = StandardScaler()

    X_train_scaled = input_scaler.fit_transform(
        X_train_reduced
    )

    X_validation_scaled = input_scaler.transform(
        X_validation_reduced
    )

    # Scaling the protein targets makes the neural-network
    # loss treat proteins with different value ranges fairly.
    target_scaler = StandardScaler()

    Y_train_scaled = target_scaler.fit_transform(
        Y_train
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

    model = CiteMLP(
        number_of_input_components=X_train_reduced.shape[1],
        number_of_proteins=Y_train.shape[1]
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
        weight_decay=0.0001
    )

    loss_function = nn.MSELoss()

    number_of_epochs = 100

    for epoch in range(1, number_of_epochs + 1):

        model.train()
        total_loss = 0.0

        for batch_inputs, batch_targets in train_loader:

            optimizer.zero_grad()

            predictions = model(
                batch_inputs
            )

            loss = loss_function(
                predictions,
                batch_targets
            )

            loss.backward()
            optimizer.step()

            total_loss += (
                loss.item()
                * batch_inputs.shape[0]
            )

        average_loss = (
            total_loss
            / len(train_dataset)
        )

        if epoch == 1 or epoch % 20 == 0:
            print(
                "CITE MLP epoch:",
                epoch,
                "training loss:",
                average_loss
            )

    model.eval()

    with torch.no_grad():
        predicted_proteins_scaled = model(
            X_validation_tensor
        ).numpy()

    predicted_proteins = (
        target_scaler.inverse_transform(
            predicted_proteins_scaled
        )
    )

    model_rmse = calculate_rmse(
        Y_validation,
        predicted_proteins
    )

    model_pearson = calculate_mean_feature_pearson(
        Y_validation,
        predicted_proteins
    )

    print("\nCITE MLP")

    print(
        "Model protein RMSE:",
        model_rmse
    )

    print(
        "Model mean protein Pearson correlation:",
        model_pearson
    )

    return (
        model,
        input_scaler,
        target_scaler,
        predicted_proteins
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
def save_pca_variance_plot(
    pca,
    title,
    output_path,
    number_to_show=20
):
    """
    Plot the variance explained by the first PCA components.
    """

    number_to_show = min(
        number_to_show,
        len(pca.explained_variance_ratio_)
    )

    component_numbers = np.arange(
        1,
        number_to_show + 1
    )

    explained_variance = (
        pca.explained_variance_ratio_[
            :number_to_show
        ]
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

    axis.grid(
        axis="y",
        alpha=0.25
    )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=220
    )

    plt.close(figure)

if __name__ == "__main__":

    train_rows, validation_rows = choose_cite_rows()

    X_train, Y_train = read_cite_rows(
        train_rows
    )

    X_validation, Y_validation = read_cite_rows(
        validation_rows
    )
    (
        X_train_reduced,
        X_validation_reduced,
        selected_genes,
        rna_pca
    ) = reduce_cite_inputs(
        X_train,
        X_validation
    )
    save_pca_scatter_plot(
        X_train_reduced,
        X_validation_reduced,
        "CITE-seq RNA PCA: training versus held-out donor",
        "cite_pca_scatter.png"
    )

    save_pca_variance_plot(
        rna_pca,
        "CITE-seq RNA PCA explained variance",
        "cite_pca_variance.png"
    )

    print("\nREDUCED CITE DATA SHAPES")

    print(
        "X_train_reduced:",
        X_train_reduced.shape
    )

    print(
        "X_validation_reduced:",
        X_validation_reduced.shape
    )

    print(
        "Number of selected RNA genes:",
        len(selected_genes)
    )

    print(
        "RNA variance retained by PCA:",
        rna_pca.explained_variance_ratio_.sum()
    )

    print("\nREAD CITE DATA SHAPES")

    print("X_train:", X_train.shape)
    print("Y_train:", Y_train.shape)

    print("X_validation:", X_validation.shape)
    print("Y_validation:", Y_validation.shape)

    print(
        "\nNonzero RNA features in first training cell:",
        np.count_nonzero(X_train[0])
    )

    print(
        "Nonzero proteins in first training cell:",
        np.count_nonzero(Y_train[0])
    )
    fit_cite_ridge_model(
        X_train_reduced,
        X_validation_reduced,
        Y_train,
        Y_validation
    )

    fit_cite_mlp_model(
        X_train_reduced,
        X_validation_reduced,
        Y_train,
        Y_validation
    )
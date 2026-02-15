import torch
import numpy as np
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import os
import random


# ---------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------
# Train / Validation Split
# ---------------------------------------------------------
def create_data_loaders(dataset, batch_size=32, test_size=0.2):
    indices = list(range(len(dataset)))
    train_idx, val_idx = train_test_split(
        indices, test_size=test_size, random_state=42
    )

    train_dataset = dataset[train_idx]
    val_dataset = dataset[val_idx]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    return train_loader, val_loader


# ---------------------------------------------------------
# Training Loop
# ---------------------------------------------------------
def train_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        out = model(batch).squeeze()
        loss = loss_fn(out, batch.y.squeeze())

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch.num_graphs

    return total_loss / len(loader.dataset)


# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------
def evaluate(model, loader, task="regression", device="cpu"):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch).squeeze()

            y_true.extend(batch.y.cpu().numpy())
            y_pred.extend(out.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if task == "classification":
        probs = torch.sigmoid(torch.tensor(y_pred)).numpy()
        auc = roc_auc_score(y_true, probs)
        return {"AUC": auc}

    elif task == "regression":
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        return {"RMSE": rmse, "MAE": mae}


# ---------------------------------------------------------
# Full Training Pipeline
# ---------------------------------------------------------
def train_model(
    model,
    dataset,
    task="regression",
    batch_size=32,
    lr=0.001,
    epochs=100,
    patience=10,
    model_save_path="best_gnn_model.pt"
):
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_loader, val_loader = create_data_loaders(dataset, batch_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if task == "classification":
        loss_fn = torch.nn.BCEWithLogitsLoss()
    else:
        loss_fn = torch.nn.MSELoss()

    best_metric = np.inf if task == "regression" else 0
    early_stop_counter = 0

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        metrics = evaluate(model, val_loader, task, device)

        print(f"\nEpoch {epoch}")
        print(f"Train Loss: {train_loss:.4f}")
        print("Validation Metrics:", metrics)

        # Early stopping logic
        if task == "regression":
            current_metric = metrics["RMSE"]
            improved = current_metric < best_metric
        else:
            current_metric = metrics["AUC"]
            improved = current_metric > best_metric

        if improved:
            best_metric = current_metric
            early_stop_counter = 0
            torch.save(model.state_dict(), model_save_path)
            print("Model improved. Saving checkpoint.")
        else:
            early_stop_counter += 1

        if early_stop_counter >= patience:
            print("Early stopping triggered.")
            break

    print("\nTraining complete.")
    print("Best validation metric:", best_metric)

    return model

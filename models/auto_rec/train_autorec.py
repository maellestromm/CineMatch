import json
import sqlite3

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from util import root_path


# ==========================================
# 1. Core Neural Network Architecture (Deep AutoRec)
# ==========================================
class DeepAutoRec(nn.Module):
    def __init__(self, num_movies):
        super(DeepAutoRec, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(num_movies, 512),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.Sigmoid()
        )
        self.decoder = nn.Sequential(
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, num_movies)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


# ==========================================
# 2. Masked Mean Squared Error (Masked MSE Loss)
# ==========================================
def masked_mse_loss(predictions, targets):
    mask = (targets != 0).float()
    error = (predictions - targets) * mask
    loss = (error ** 2).sum() / (mask.sum() + 1e-8)
    return loss


# ==========================================
# 3. Data Loading and Dimensional Alignment
# ==========================================
def load_and_align_data(train_db=root_path() / "data/train_model.db", test_db=root_path() / "data/test_eval.db"):
    print("Reading and aligning physically isolated databases...")

    # --- Load Training Set ---
    conn_train = sqlite3.connect(train_db)
    df_train = pd.read_sql_query(
        "SELECT user_username, movie_slug, rating FROM reviews WHERE rating != 'None' AND rating IS NOT NULL",
        conn_train)
    conn_train.close()
    df_train['rating'] = df_train['rating'].astype(float)
    matrix_train = df_train.pivot(index='user_username', columns='movie_slug', values='rating').fillna(0)

    # Establish global standard movie dimensions
    movie_slugs = matrix_train.columns.tolist()
    num_movies = len(movie_slugs)

    # --- Load Test Set ---
    conn_test = sqlite3.connect(test_db)
    df_test = pd.read_sql_query(
        "SELECT user_username, movie_slug, rating FROM reviews WHERE rating != 'None' AND rating IS NOT NULL",
        conn_test)
    conn_test.close()
    df_test['rating'] = df_test['rating'].astype(float)
    matrix_test = df_test.pivot(index='user_username', columns='movie_slug', values='rating').fillna(0)

    # Force test set columns to match train set
    matrix_test = matrix_test.reindex(columns=movie_slugs, fill_value=0.0)

    print("Data matrix construction complete!")
    print(f"   Train : {matrix_train.shape[0]} Users x {num_movies} Movies")
    print(f"   Test  : {matrix_test.shape[0]} Users x {num_movies} Movies")

    # Convert to PyTorch tensors
    train_tensor = torch.FloatTensor(matrix_train.values.copy())
    test_tensor = torch.FloatTensor(matrix_test.values.copy())

    return train_tensor, test_tensor, movie_slugs


# ==========================================
# 4. Training Loop with Early Stopping
# ==========================================
def train_model(epochs=100, batch_size=256, lr=0.005, patience=8):
    train_tensor, test_tensor, movie_slugs = load_and_align_data()
    num_movies = len(movie_slugs)

    # Save dictionary for future inference
    with open(root_path() / "data/movie_dictionary.json", "w", encoding="utf-8") as f:
        json.dump(movie_slugs, f)

    train_dataset = TensorDataset(train_tensor, train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_x = test_tensor
    test_y = test_tensor

    model = DeepAutoRec(num_movies=num_movies)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []

    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_epoch = 0

    print("\nStarting deep learning training (Monitoring Test Loss to prevent overfitting)...")

    for epoch in range(epochs):
        # --- Training Phase ---
        model.train()
        total_train_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = masked_mse_loss(preds, batch_y)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # --- Validation Phase ---
        model.eval()
        with torch.no_grad():
            test_preds = model(test_x)
            val_loss = masked_mse_loss(test_preds, test_y).item()
            val_losses.append(val_loss)

        print(f"   Epoch [{epoch + 1:03d}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Test Loss: {val_loss:.4f}")

        # --- Early Stopping Logic ---
        if val_loss < best_val_loss - 0.0005:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_epoch = epoch + 1
            # Save weights only when a new record is set
            torch.save(model.state_dict(), root_path() / "data/autorec_best_weights.pth")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"\nEarly Stopping triggered! Test error hasn't improved for {patience} consecutive epochs.")
            print(f"Best model remains at Epoch {best_epoch}, Test Loss: {best_val_loss:.4f}")
            break

    # ==========================================
    # 5. Generate Dual-line Loss Curve
    # ==========================================
    plt.figure(figsize=(10, 6))
    actual_epochs = len(train_losses)

    plt.plot(range(1, actual_epochs + 1), train_losses, label='Train Loss (5457 Users)', color='#1f77b4', linewidth=2)
    plt.plot(range(1, actual_epochs + 1), val_losses, label='Test/Val Loss (606 Users)', color='#ff7f0e', linewidth=2,
             linestyle='--')

    plt.axvline(x=best_epoch, color='r', linestyle=':', label=f'Best Epoch ({best_epoch})')
    plt.scatter(best_epoch, best_val_loss, color='red', zorder=5)

    plt.title('Deep AutoRec Generalization: Train vs Test Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Masked MSE Loss', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.savefig('train_vs_test_loss.png', dpi=300, bbox_inches='tight')
    print("\nDual-track Loss curve saved as train_vs_test_loss.png")


if __name__ == "__main__":
    train_model(epochs=150, batch_size=256, lr=0.0005, patience=15)

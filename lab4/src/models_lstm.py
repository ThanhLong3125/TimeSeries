import os
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from typing import Dict, Any, List, Optional
import logging

class LSTMNetwork(nn.Module):
    def __init__(self, input_size: int = 4, hidden_dim1: int = 64, hidden_dim2: int = 32, output_dim: int = 1):
        super(LSTMNetwork, self).__init__()
        # LSTM(64) -> LSTM(32) -> Dense(1) theo đúng đề bài
        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=hidden_dim1, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=hidden_dim1, hidden_size=hidden_dim2, batch_first=True)
        self.fc = nn.Linear(hidden_dim2, output_dim)

    def forward(self, x):
        # x shape: (batch_size, 24, 4)
        out1, _ = self.lstm1(x)         # out1 shape: (batch_size, 24, 64)
        out2, _ = self.lstm2(out1)       # out2 shape: (batch_size, 24, 32)
        last_timestep = out2[:, -1, :]  # lấy vector trạng thái của bước thời gian cuối cùng t
        pred = self.fc(last_timestep)   # shape: (batch_size, 1)
        return pred.squeeze(-1)


class LSTMModelWrapper:
    def __init__(
        self,
        input_size: int = 4,
        hidden_dim1: int = 64,
        hidden_dim2: int = 32,
        output_dim: int = 1,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        num_epochs: int = 50,
        early_stopping_patience: int = 10,
        logger: Optional[logging.Logger] = None
    ):
        self.input_size = input_size
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.early_stopping_patience = early_stopping_patience
        self.logger = logger

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = LSTMNetwork(input_size, hidden_dim1, hidden_dim2, output_dim).to(self.device)
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float("inf")

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray):
        train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
        val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        patience_counter = 0
        best_state = None

        if self.logger:
            self.logger.info(f"Bắt đầu huấn luyện LSTM trên {self.device} (Arch: LSTM(64)->LSTM(32)->Dense(1))...")

        for epoch in range(1, self.num_epochs + 1):
            self.model.train()
            total_train_loss = 0.0
            for bx, by in train_loader:
                bx, by = bx.to(self.device), by.to(self.device)
                optimizer.zero_grad()
                pred = self.model(bx)
                loss = criterion(pred, by)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item() * len(bx)

            avg_train_loss = total_train_loss / len(X_train)
            self.train_losses.append(avg_train_loss)

            # Validation
            self.model.eval()
            total_val_loss = 0.0
            with torch.no_grad():
                for vx, vy in val_loader:
                    vx, vy = vx.to(self.device), vy.to(self.device)
                    v_pred = self.model(vx)
                    v_loss = criterion(v_pred, vy)
                    total_val_loss += v_loss.item() * len(vx)

            avg_val_loss = total_val_loss / len(X_val)
            self.val_losses.append(avg_val_loss)

            if epoch % 5 == 0 or epoch == 1:
                if self.logger:
                    self.logger.info(f"Epoch {epoch:2d}/{self.num_epochs:2d} - Train MSE: {avg_train_loss:.5f} | Val MSE: {avg_val_loss:.5f}")

            # Early stopping check
            if avg_val_loss < self.best_val_loss:
                self.best_val_loss = avg_val_loss
                best_state = self.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping_patience:
                    if self.logger:
                        self.logger.info(f"Early stopping kích hoạt tại epoch {epoch}. Best Val Loss: {self.best_val_loss:.5f}")
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        loader = DataLoader(TensorDataset(torch.from_numpy(X)), batch_size=self.batch_size, shuffle=False)
        preds = []
        with torch.no_grad():
            for (bx,) in loader:
                bx = bx.to(self.device)
                pred = self.model(bx)
                preds.append(pred.cpu().numpy())
        return np.concatenate(preds)

    def save(self, file_path: str):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "config": {
                "input_size": self.input_size,
                "hidden_dim1": self.hidden_dim1,
                "hidden_dim2": self.hidden_dim2,
                "output_dim": self.output_dim,
                "learning_rate": self.learning_rate
            },
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "best_val_loss": self.best_val_loss
        }, file_path)

    @classmethod
    def load(cls, file_path: str, logger: Optional[logging.Logger] = None):
        checkpoint = torch.load(file_path, map_location="cpu", weights_only=False)
        wrapper = cls(**checkpoint["config"], logger=logger)
        wrapper.model.load_state_dict(checkpoint["model_state_dict"])
        wrapper.train_losses = checkpoint.get("train_losses", [])
        wrapper.val_losses = checkpoint.get("val_losses", [])
        wrapper.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        return wrapper

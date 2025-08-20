import os
import argparse
import shutil
import time
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from pytorch.models import SmallModel
from pytorch.generate_data import generate_batch_fixed


def numpy_to_tensor(images: np.ndarray, targets: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
    # images: B x H x W x 6 -> B x 6 x H x W
    images_t = torch.from_numpy(images.transpose(0, 3, 1, 2)).float()
    # targets: B x N x M x 2
    targets_t = torch.from_numpy(targets.transpose(0, 3, 1, 2)).float()
    return images_t, targets_t


def evaluate(model: nn.Module, device: torch.device, setting=(80, 112, 10, 14)) -> float:
    model.eval()
    with torch.no_grad():
        X_val, Y_val = generate_batch_fixed(batch_size=1000, setting=setting)
        images_t, targets_t = numpy_to_tensor(X_val, Y_val)
        images_t = images_t.to(device)
        targets_t = targets_t.to(device)
        outputs = model(images_t)
        loss = nn.functional.mse_loss(outputs, targets_t).item()
    return float(loss)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--prefix", default="torch_small")
    parser.add_argument("-lr", "--lr", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--save-dir", default="models")
    args = parser.parse_args()

    prefix = args.prefix
    lr = args.lr
    epochs = args.epochs
    steps_per_epoch = args.steps
    batch_size = args.batch_size
    save_root = args.save_dir

    os.makedirs(os.path.join(save_root, prefix), exist_ok=True)
    # Archive training script and data generator for reproducibility
    shutil.copy(__file__, os.path.join(save_root, prefix, "train.py"))
    src_gen = os.path.join(os.path.dirname(__file__), "generate_data.py")
    shutil.copy(src_gen, os.path.join(save_root, prefix, "generate_data.py"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    criterion = nn.MSELoss()

    best_loss = float("inf")
    val_setting = (80, 112, 10, 14)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        start_time = time.time()
        for step in range(steps_per_epoch):
            X_batch, Y_batch = generate_batch_fixed(batch_size=batch_size, setting=val_setting)
            images_t, targets_t = numpy_to_tensor(X_batch, Y_batch)
            images_t = images_t.to(device)
            targets_t = targets_t.to(device)

            optimizer.zero_grad()
            outputs = model(images_t)
            loss = criterion(outputs, targets_t)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())

        avg_loss = running_loss / steps_per_epoch
        val_loss = evaluate(model, device, setting=val_setting)
        elapsed = time.time() - start_time
        print(f"epoch {epoch}: train_loss={avg_loss:.6f}, val_loss={val_loss:.6f}, time={elapsed:.1f}s")

        if val_loss < best_loss:
            best_loss = val_loss
            save_path = os.path.join(save_root, prefix, f"tracking_{epoch:03d}_{val_loss:.3f}.pt")
            torch.save({
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
            }, save_path)


if __name__ == "__main__":
    main()


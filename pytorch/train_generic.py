import os
import argparse
import shutil
import time
from typing import Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from pytorch.models import GenericAE
from pytorch.generate_data import generate_batch_generic


def numpy_to_tensor(images: np.ndarray, targets: List[np.ndarray]):
    # images: B x H x W x 8 -> B x 8 x H x W
    images_t = torch.from_numpy(images.transpose(0, 3, 1, 2)).float()
    # targets: list of levels with shapes:
    #   L0: B x H x W x 2
    #   L1: B x H/2 x W/2 x 2
    #   L2: B x H/4 x W/4 x 2
    #   L3: B x H/8 x W/8 x 2
    #   L4: B x H/16 x W/16 x 2
    targets_t = [torch.from_numpy(t.transpose(0, 3, 1, 2)).float() for t in targets]
    return images_t, targets_t


def evaluate(model: nn.Module, device: torch.device, batch_size: int = 1000) -> float:
    model.eval()
    with torch.no_grad():
        X_val, Y_val_list = generate_batch_generic(batch_size=batch_size)
        images_t, targets_t = numpy_to_tensor(X_val, Y_val_list)
        images_t = images_t.to(device)
        targets_t = [t.to(device) for t in targets_t]
        outputs = model(images_t)
        loss = 0.0
        for out, tgt in zip(outputs, targets_t):
            loss += nn.functional.mse_loss(out, tgt)
        return float(loss.item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--prefix", default="torch_generic")
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
    shutil.copy(__file__, os.path.join(save_root, prefix, "train_generic.py"))
    src_gen = os.path.join(os.path.dirname(__file__), "generate_data.py")
    shutil.copy(src_gen, os.path.join(save_root, prefix, "generate_data.py"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GenericAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))

    best_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        start_time = time.time()
        for step in range(steps_per_epoch):
            X_batch, Y_batch_list = generate_batch_generic(batch_size=batch_size)
            images_t, targets_t = numpy_to_tensor(X_batch, Y_batch_list)
            images_t = images_t.to(device)
            targets_t = [t.to(device) for t in targets_t]

            optimizer.zero_grad()
            outputs = model(images_t)
            loss = 0.0
            for out, tgt in zip(outputs, targets_t):
                loss = loss + nn.functional.mse_loss(out, tgt)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())

        avg_loss = running_loss / steps_per_epoch
        val_loss = evaluate(model, device, batch_size=1000)
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


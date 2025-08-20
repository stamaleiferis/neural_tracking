## PyTorch Training

This folder contains PyTorch equivalents of the original TensorFlow/Keras training scripts. You can train both the fixed-grid model and the generic model.

### Setup

- Python 3.8+
- Install dependencies:

```
pip install torch torchvision opencv-python numpy
```

Optional for visualization/debugging:

```
pip install matplotlib
```

### Train the fixed-grid model (small)

This matches `train.py` and learns marker displacements on a fixed 10x14 grid from 80x112 inputs.

```
python pytorch/train.py -p torch_small -lr 1e-5 --epochs 100 --steps 2000 --batch-size 32
```

Arguments:

- `-p/--prefix`: model save subfolder under `models/`
- `-lr/--lr`: learning rate
- `--epochs`: number of epochs
- `--steps`: steps per epoch (each step pulls a fresh synthetic batch)
- `--batch-size`: synthetic batch size

### Train the generic model (encoder-decoder)

This matches `train_generic.py` and learns a dense flow field at multiple scales from variable-sized inputs and marker grids.

```
python pytorch/train_generic.py -p torch_generic -lr 1e-5 --epochs 100 --steps 2000 --batch-size 32
```

Notes:

- Models are saved to `models/<prefix>/tracking_XXX_LOSS.pt` whenever validation loss improves.
- Scripts are self-contained and generate synthetic training data on the fly (no external datasets required).
- Run from the repository root so relative imports work.


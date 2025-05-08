
# Cubism-to-Nature CycleGAN

This project implements a CycleGAN model using PyTorch to translate images between two domains: Cubism artworks and nature photographs.

# Results
![Nautre to Cubism](cycle-gan/output_images/epoch_96_X_to_Y.png)

![Cubism to Nature](cycle-gan/output_images/epoch_96_Y_to_X.png)


## Project Structure

```
.
├── generator.py          # Generator architecture
├── discriminator.py      # Discriminator architecture
├── data_load.py          # Custom dataset and transforms
├── main.py               # Training and hyperparameter optimization with Optuna
├── best_parms.json       # Best params saved from hyperband parameter tuning
├── environment.yml       # Conda packages used for reproducing the environment to run the files
├── input_photos/         # Contains subfolders 'cubism' and 'nature' with input images
└── output_images/        # Generated outputs during training

```

## Key Components

### Generator

- Implements an architecture with:
  - Initial convolutional block (ReflectionPad2d, Conv2d, InstanceNorm2d, ReLU)
  - Downsampling (Conv2d layers)
  - 9 Residual Blocks
  - Upsampling (ConvTranspose2d)
  - Final output block with Tanh activation

### Discriminator

- Implements an architecture with:
  - Sequential Conv2d blocks
  - InstanceNorm2d and LeakyReLU
  - Final Sigmoid layer for real/fake classification

### Data Preparation

- Downloads and stores Cubism images from the WikiArt dataset and nature images from the `mertcobanov/nature-dataset`.
- Stored in `input_photos/cubism/` and `input_photos/nature/`.

### Dataset

- Uses a custom `ImageTransform` dataset class for:
  - Resizing, cropping, flipping, and normalization
  - Synchronized transformations for photo-nature pairs

### Training

- Implements full CycleGAN training loop
- Uses Optuna for hyperparameter tuning (`lr_G`, `lr_D`, `batch_size`, etc.)
- Trains for 100 epochs using best hyperparameters

### Visualization

- Saves image grids comparing real and generated images every 5 epochs to `output_images/`

## How to Run

1. **Install Dependencies**

```bash
conda env create -f environment.yml
conda activate cubism2nature
```

2. **Run the Script**

```bash
python main.py
```

This will download the datasets, tune hyperparameters with Optuna, and begin training.

## Output

- `output_images/epoch_{n}_X_to_Y.png`: Cubism to Nature results
- `output_images/epoch_{n}_Y_to_X.png`: Nature to Cubism results
- `best_params.json`: Best hyperparameters found

## Notes

- Supports multiple GPUs via `torch.nn.DataParallel`.
- Set `training=False` during hyperparameter tuning to skip saving intermediate results.

---

Inspired by the original [CycleGAN paper](https://arxiv.org/abs/1703.10593) by Zhu et al.

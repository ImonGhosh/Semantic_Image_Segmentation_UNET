# Image Segmentation with U-Net

- A clean, reproducible PyTorch pipeline for binary image segmentation of cars using a U-Net.
- Trains on RGB images and corresponding binary masks and reports Dice score + pixel accuracy.
- Best run achieved: Dice = 0.9837981462478638 (near-perfect masks on the Carvana split).

## Project structure :

```bash
.
├─ train.ipynb               # end-to-end training/eval notebook
├─ model.py                  # U-Net model (encoder–decoder + skip connections)
├─ utils.py                  # loaders, checkpointing, metrics, save preds
├─ dataset.py                # CarvanaDataset: RGB image + binary mask
└─ data/
   ├─ train_images/          # e.g., xxx.jpg
   ├─ train_masks/           # e.g., xxx_mask.gif  (binary 0/255)
   ├─ val_images/
   └─ val_masks/
```

## Dataset

- Source: Carvana Image Masking Challenge (cars in various poses). (Link - https://www.kaggle.com/competitions/carvana-image-masking-challenge/overview)
- Inputs: color images of cars; targets: binary masks (car = 1, background = 0).
- File naming: for an image name.jpg, the mask is name_mask.gif (see dataset.py).
- Splits: data/train_* and data/val_* directories (you provide the splits).

## Preprocessing & Augmentation

- Resize to 240×160 (W×H) for training and validation.
- Train-time augments: Rotate(±35°), HorizontalFlip(0.5), VerticalFlip(0.1).
- Normalization to mean= [0,0,0], std= [1,1,1].
- Masks are loaded as float (0/1) with 255 → 1 conversion.
- Implemented with Albumentations; converted to tensors via ToTensorV2.

## Key hyperparameters (defaults in train.ipynb)

Optimizer: Adam, lr = 1e-4
Batch size: 16
Epochs: 3 (increase for stronger results)
Image size: 160×240 (H×W stored as constants)
Loss: BCEWithLogitsLoss (sigmoid applied only for metrics/saving)
AMP: mixed precision with torch.cuda.amp (GradScaler enabled)
For multi-class segmentation: set out_channels > 1 in UNET and use CrossEntropyLoss (drop per-channel sigmoid; use logits directly).

## Requirements
```bash
python>=3.9
torch>=2.0
torchvision
albumentations
tqdm
Pillow
numpy

pip install torch torchvision albumentations tqdm pillow numpy
```

## Getting started

- Prepare data under data/ as shown above (respect the mask naming).

- (Optional) Adjust constants in train.ipynb:
  TRAIN_IMG_DIR, TRAIN_MASK_DIR, VAL_IMG_DIR, VAL_MASK_DIR
  IMAGE_HEIGHT, IMAGE_WIDTH, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE

- Run the notebook cells to train, evaluate, and save predictions.

## Training & Evaluation Metric

Training loop: train_fn(...) with AMP, progress via tqdm.

Metrics (utils.check_accuracy):

1. Pixel accuracy

2. Dice score (averaged over validation loader)

Best result observed on the provided split:
Dice Score = 0.9837981462478638

## Checkpoints & Predictions

After each epoch, a checkpoint is saved as my_checkpoint.pth.tar (state_dict + optimizer).

To resume: set LOAD_MODEL = True in train.ipynb.

Validation predictions are written to saved_images/ as:
pred_{idx}.png → predicted binary mask
{idx}.png → ground-truth mask

## Customization tips

- Resolution: increase IMAGE_HEIGHT/IMAGE_WIDTH for finer masks (trade-off: memory/compute).

- Augmentations: tune Albumentations to match your domain.

- Class imbalance: consider Dice loss / BCE+Dice hybrid if foreground is sparse.

- Longer training: raise NUM_EPOCHS, use LR schedules, or try weight decay.

- Backbone width: change features=[...] in UNET for lighter/heavier models.

## Things to keep in mind

- Masks are thresholded at 0.5 after sigmoid for metrics/saving.
- utils.save_predictions_as_imgs expects a folder (default saved_images/) to exist or be creatable.
- NUM_WORKERS=0 is notebook-friendly; increase on Linux for speed.

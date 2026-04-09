# Project 2: I2I Translation and Input Space UDA

NTU Advanced Computer Vision — Project 2

Compares image style transfer in spatial vs. spectral space and benchmarks 5 unsupervised domain adaptation (UDA) methods across 5 dataset pairs.

## Tasks

**Task I — Style Transfer Comparison**
- Spatial CycleGAN (standard pixel-space translation)
- Spectral CycleGAN (CycleGAN on low-frequency FFT components only, recombined with original high-frequency at inference)

**Task II — UDA Benchmark**

| Method | Description |
|---|---|
| Source-only | Train on source, test on target directly (baseline) |
| CycleGAN | Train classifier on CycleGAN-translated source images |
| Spectral CycleGAN | Train classifier on spectral CycleGAN-translated source images |
| CyCADA | CycleGAN + task-consistency loss (Hoffman et al., ICML 2018) |
| FDA | Fourier Domain Adaptation — swap low-freq amplitude bands (Yang et al., CVPR 2020) |

## Datasets

| Pair | Source → Target | Classes | Classifier |
|---|---|---|---|
| 1 | MNIST → USPS | 10 | LeNet-5 |
| 2 | SVHN → MNIST | 10 | LeNet-5 |
| 3 | Amazon → Webcam (Office-31) | 31 | ResNet-50 |
| 4 | Art → Real World (Office-Home) | 65 | ResNet-50 |
| 5 | Photo → Sketch (PACS) | 7 | ResNet-50 |

MNIST, USPS, and SVHN are auto-downloaded by torchvision. All three object datasets are included in `data/` (Office-Home is sampled to 50% per class to fit GitHub size limits).

## Project Structure

```
Project2/
├── notebooks/
│   ├── mnist_usps.ipynb          # MNIST → USPS
│   ├── svhn_mnist.ipynb          # SVHN → MNIST
│   ├── amazon_webcam.ipynb       # Amazon → Webcam
│   ├── art_realworld.ipynb       # Art → Real World
│   └── photo_sketch.ipynb        # Photo → Sketch
├── utils/
│   ├── fft_utils.py              # FFT mask, FDA transfer, spectral decompose/reconstruct
│   ├── data_utils.py             # Dataset loaders for all 5 pairs
│   ├── classifiers.py            # LeNet-5 and ResNet-50 with train/eval loops
│   ├── cyclegan_wrapper.py       # CycleGAN train/test command generators
│   ├── spectral_cyclegan.py      # Low/high freq decomposition and reconstruction
│   ├── cycada.py                 # CyCADA: CycleGAN + task loss
│   ├── viz_utils.py              # Image grids and results tables
│   └── eval_utils.py             # Save/load results as JSON
└── data/
    ├── office31/                 # Included (87 MB)
    ├── office_home/              # Included, sampled 50% (470 MB)
    └── pacs/                     # Included (202 MB)
```

## Setup

### Option A: Google Colab

1. Open any notebook in Colab and **Run All**. The first cell will automatically:
   - `git clone` this repo to `/content/Project2`
   - Clone the [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) repo
   - Mount Google Drive (for saving results only)

2. Set `FAST_TEST = True` first to verify everything works (~5 min), then `FAST_TEST = False` for full training.

### Option B: Server / Local Machine

1. **Clone the repo and install dependencies:**

```bash
git clone https://github.com/sophie99zyq/NTU-AdvancedCV-Project2.git
cd NTU-AdvancedCV-Project2
pip install torch torchvision matplotlib visdom dominate pillow
git clone https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix.git
```

2. **Modify the first cell of each notebook** before running:

```python
# Remove or comment out these Colab-specific lines:
# from google.colab import drive
# drive.mount('/content/drive')

# Change paths to match your local setup:
import sys
sys.path.insert(0, '/path/to/NTU-AdvancedCV-Project2')  # project root

# Change SAVE_DIR to a local path (originally saves to Google Drive):
SAVE_DIR = '/path/to/NTU-AdvancedCV-Project2/results/xxx'
```

3. **Path substitution cheat sheet** — replace these Colab paths throughout the notebook:

| Colab path | Server equivalent |
|---|---|
| `/content/Project2` | your cloned repo root |
| `/content/drive/MyDrive/Project2/results/...` | `<repo>/results/...` |
| `/content/cyclegan_data/...` | `<repo>/cyclegan_data/...` (or any local dir) |
| `/content/spectral_data/...` | `<repo>/spectral_data/...` |
| `/content/spectral_reconstructed/...` | `<repo>/spectral_reconstructed/...` |

4. **Run notebooks** with Jupyter:

```bash
pip install jupyter
jupyter notebook notebooks/
```

Or convert to a Python script and run directly:

```bash
jupyter nbconvert --to script notebooks/mnist_usps.ipynb
python notebooks/mnist_usps.py
```

### Notes

- Set `FAST_TEST = True` first to verify everything works (~5 min)
- Then set `FAST_TEST = False` for full training
- GPU is strongly recommended — training CycleGAN on CPU is very slow
- MNIST, USPS, and SVHN are auto-downloaded by torchvision; the three object datasets are in `data/`

Each notebook produces:
- Task I: side-by-side style transfer visualizations
- Task II: accuracy table for all 5 UDA methods

## References

1. Zhu et al., *Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks*, ICCV 2017
2. Hoffman et al., *CyCADA: Cycle-Consistent Adversarial Domain Adaptation*, ICML 2018
3. Yang et al., *FDA: Fourier Domain Adaptation for Semantic Segmentation*, CVPR 2020

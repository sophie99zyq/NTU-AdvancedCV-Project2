import os
import torch
from torchvision.utils import save_image
from utils.fft_utils import spectral_decompose, spectral_reconstruct


def save_low_freq_images(image_dir, output_dir, beta=0.03, mode='RGB'):
    """Decompose all images into low-freq and high-freq, save to subdirs.
    Args:
        mode: 'RGB' for color images, 'L' for grayscale
    """
    from torchvision import transforms
    from PIL import Image

    low_dir = os.path.join(output_dir, 'low')
    high_dir = os.path.join(output_dir, 'high')
    os.makedirs(low_dir, exist_ok=True)
    os.makedirs(high_dir, exist_ok=True)

    to_tensor = transforms.ToTensor()

    for fname in sorted(os.listdir(image_dir)):
        if not fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            continue
        img = Image.open(os.path.join(image_dir, fname)).convert(mode)
        img_t = to_tensor(img).unsqueeze(0)
        low, high = spectral_decompose(img_t, beta=beta)
        save_image(low.squeeze(0), os.path.join(low_dir, fname))
        save_image(high.squeeze(0), os.path.join(high_dir, fname))

    print(f"Saved {len(os.listdir(low_dir))} low-freq images to {low_dir}")
    return low_dir, high_dir


def reconstruct_from_translated_low(translated_low_dir, original_high_dir, output_dir):
    """Reconstruct full images from translated low-freq + original high-freq.

    Pairs files by sorted order (CycleGAN output names like 00000_fake_B.png
    may differ from original names like 00000.png).
    """
    from torchvision import transforms
    from PIL import Image

    os.makedirs(output_dir, exist_ok=True)
    to_tensor = transforms.ToTensor()

    def get_image_files(d):
        return sorted([f for f in os.listdir(d) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])

    # Filter translated images: only keep _fake_B (A->B translation) if CycleGAN output
    low_files = get_image_files(translated_low_dir)
    fake_b_files = [f for f in low_files if '_fake_B' in f]
    if fake_b_files:
        low_files = fake_b_files  # use only fake_B images from CycleGAN

    high_files = get_image_files(original_high_dir)

    n = min(len(low_files), len(high_files))
    for i in range(n):
        low = to_tensor(Image.open(os.path.join(translated_low_dir, low_files[i])).convert('RGB')).unsqueeze(0)
        high = to_tensor(Image.open(os.path.join(original_high_dir, high_files[i])).convert('RGB')).unsqueeze(0)
        recon = spectral_reconstruct(low, high)
        save_image(recon.squeeze(0), os.path.join(output_dir, high_files[i]))

    print(f"Reconstructed {n} images to {output_dir}")
    return output_dir

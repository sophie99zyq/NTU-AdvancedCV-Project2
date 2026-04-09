import torch
import numpy as np


def create_low_freq_mask(h, w, beta):
    """Create circular low-frequency mask for FFT-shifted spectrum.
    Args: h (height), w (width), beta (ratio of radius to min(h,w), e.g. 0.01-0.05)
    Returns: (h, w) binary float tensor
    """
    cy, cx = h // 2, w // 2
    radius = max(1, int(min(h, w) * beta))
    y = torch.arange(h).unsqueeze(1) - cy
    x = torch.arange(w).unsqueeze(0) - cx
    mask = (x ** 2 + y ** 2) <= radius ** 2
    return mask.float()


def fda_transfer(source_img, target_img, beta=0.01):
    """FDA: replace low-freq amplitude of source with target's in spectral domain.
    Args: source_img (B,C,H,W), target_img (B,C,H,W), beta (bandwidth ratio)
    Returns: transferred (B,C,H,W) with target's low-freq style
    """
    src_fft = torch.fft.fft2(source_img, dim=(-2, -1))
    tgt_fft = torch.fft.fft2(target_img, dim=(-2, -1))
    src_fft_shifted = torch.fft.fftshift(src_fft, dim=(-2, -1))
    tgt_fft_shifted = torch.fft.fftshift(tgt_fft, dim=(-2, -1))

    _, _, h, w = source_img.shape
    mask = create_low_freq_mask(h, w, beta).to(source_img.device)
    mask = mask.unsqueeze(0).unsqueeze(0)

    src_amp = torch.abs(src_fft_shifted)
    src_phase = torch.angle(src_fft_shifted)
    tgt_amp = torch.abs(tgt_fft_shifted)

    new_amp = src_amp * (1 - mask) + tgt_amp * mask
    new_fft_shifted = new_amp * torch.exp(1j * src_phase)
    new_fft = torch.fft.ifftshift(new_fft_shifted, dim=(-2, -1))
    transferred = torch.fft.ifft2(new_fft, dim=(-2, -1)).real
    return transferred.clamp(0, 1)


def spectral_decompose(img, beta=0.03):
    """Decompose image into low-freq and high-freq via FFT.
    Args: img (B,C,H,W), beta (cutoff ratio)
    Returns: (low_freq, high_freq) both (B,C,H,W) in spatial domain
    """
    fft = torch.fft.fft2(img, dim=(-2, -1))
    fft_shifted = torch.fft.fftshift(fft, dim=(-2, -1))
    _, _, h, w = img.shape
    mask = create_low_freq_mask(h, w, beta).to(img.device)
    mask = mask.unsqueeze(0).unsqueeze(0)
    low_fft = fft_shifted * mask
    high_fft = fft_shifted * (1 - mask)
    low_freq = torch.fft.ifft2(torch.fft.ifftshift(low_fft, dim=(-2, -1)), dim=(-2, -1)).real
    high_freq = torch.fft.ifft2(torch.fft.ifftshift(high_fft, dim=(-2, -1)), dim=(-2, -1)).real
    return low_freq, high_freq


def spectral_reconstruct(low_freq, high_freq):
    """Reconstruct from low + high freq components. Returns (B,C,H,W)."""
    return (low_freq + high_freq).clamp(0, 1)

import numpy as np
import matplotlib.pyplot as plt


def plot_digit(image, label):
    """Plot a single digit"""
    plt.figure(figsize=(6, 6))
    plt.imshow(image, cmap='gray')
    plt.title(f'Digit: {label}')
    plt.colorbar(label='Pixel Value')
    plt.show()


def plot_multiple_digits(images, labels, n=10):
    """Plot multiple digits"""
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    axes = axes.ravel()

    for i in range(min(n, len(images))):
        axes[i].imshow(images[i], cmap='gray')
        axes[i].set_title(f'Label: {labels[i]}')
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()


def plot_rgb_image(image):
    """Plot an RGB image"""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # Full image
    axes[0].imshow(image)
    axes[0].set_title('RGB Image')
    axes[0].axis('off')

    # R channel
    axes[1].imshow(image[:, :, 0], cmap='Reds')
    axes[1].set_title('Red Channel')
    axes[1].axis('off')

    # G channel
    axes[2].imshow(image[:, :, 1], cmap='Greens')
    axes[2].set_title('Green Channel')
    axes[2].axis('off')

    # B channel
    axes[3].imshow(image[:, :, 2], cmap='Blues')
    axes[3].set_title('Blue Channel')
    axes[3].axis('off')

    plt.tight_layout()
    plt.show()


def plot_original_vs_normalized(original, normalized, label):
    """Compare original and normalized images"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Original
    im1 = axes[0].imshow(original, cmap='gray', vmin=0, vmax=15)
    axes[0].set_title(f'Original (0-15)\nLabel: {label}')
    plt.colorbar(im1, ax=axes[0])

    # Normalized
    im2 = axes[1].imshow(normalized, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title(f'Normalized (0-1)\nLabel: {label}')
    plt.colorbar(im2, ax=axes[1])

    plt.tight_layout()
    plt.show()


def plot_label_distribution(labels):
    """Plot distribution of labels in dataset"""
    unique, counts = np.unique(labels, return_counts=True)

    plt.figure(figsize=(10, 6))
    plt.bar(unique, counts, alpha=0.7)
    plt.xlabel('Digit')
    plt.ylabel('Count')
    plt.title('Distribution of Digits in Dataset')
    plt.xticks(unique)
    plt.grid(True, alpha=0.3, axis='y')
    plt.show()

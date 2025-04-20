import os
from pathlib import Path

from PIL import Image


def is_valid_image(file_path):
    """Check if a file is a valid image"""
    supported_formats = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}

    # Check file extension
    if Path(file_path).suffix.lower() not in supported_formats:
        return False

    # Try to open the image
    try:
        img = Image.open(file_path)
        img.verify()
        return True
    except Exception:
        return False


def scan_image_directory(directory_path, recursive=True):
    """Scan a directory for images

    Args:
        directory_path: Path to scan
        recursive: Whether to scan subdirectories

    Returns:
        list: List of valid image paths
    """
    directory = Path(directory_path)

    if recursive:
        all_files = list(directory.glob("**/*"))
    else:
        all_files = list(directory.glob("*"))

    # Filter for valid images
    image_files = [f for f in all_files if is_valid_image(f)]

    return image_files


def get_image_stats(image_path):
    """Get detailed image statistics

    Args:
        image_path: Path to image

    Returns:
        dict: Dictionary of image stats
    """
    try:
        img = Image.open(image_path)

        stats = {
            "filename": Path(image_path).name,
            "path": str(image_path),
            "width": img.width,
            "height": img.height,
            "format": img.format,
            "mode": img.mode,
            "size_kb": os.path.getsize(image_path) / 1024,
            "aspect_ratio": img.width / img.height if img.height > 0 else 0,
        }

        return stats
    except Exception as e:
        print(f"Error analyzing image {image_path}: {e}")
        return None

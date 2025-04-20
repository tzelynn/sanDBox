import numpy as np
import torch
from PIL import Image
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor


class DINOv2Embedder:
    """Class for generating image embeddings using DINOv2"""

    def __init__(self, model_size="base", device=None):
        """Initialize the DINOv2 model

        Args:
            model_size (str): One of 'small', 'base', 'large', or 'giant'
            device (str): Device to run the model on ('cuda' or 'cpu')
        """
        self.model_size = model_size
        self.device = (
            device
            if device is not None
            else "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = self._load_model()
        self.transform = self._get_transform()

    def _load_model(self):
        """Load DINOv2 model"""
        model_mapping = {
            "small": "dinov2_vits14",
            "base": "dinov2_vitb14",
            "large": "dinov2_vitl14",
            "giant": "dinov2_vitg14",
        }
        model_name = model_mapping.get(self.model_size, "dinov2_vitb14")

        print(f"Loading DINOv2 model: {model_name}")

        # Use HTTPS URL to avoid Git issues with PyTorch Hub
        model = torch.hub.load(
            "facebookresearch/dinov2",
            model_name,
            source="github",
            force_reload=False,
            trust_repo=True,
        )
        model.to(self.device)
        model.eval()
        return model

    def _get_transform(self):
        """Get image transforms for DINOv2"""
        transform = Compose(
            [
                Resize(256),
                CenterCrop(224),
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        return transform

    def get_embedding_dimension(self):
        """Get the embedding dimension for the selected model size"""
        dimensions = {"small": 384, "base": 768, "large": 1024, "giant": 1536}
        return dimensions.get(self.model_size, 768)

    def get_embedding(self, image_path):
        """Generate embedding for an image

        Args:
            image_path: Path to the image file

        Returns:
            list: Normalized embedding vector
        """
        try:
            # Load and preprocess image
            img = Image.open(image_path).convert("RGB")
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)

            # Generate embedding
            with torch.no_grad():
                embedding = self.model(img_tensor)

            # Normalize embedding
            embedding = embedding.cpu().numpy()[0]
            embedding = embedding / np.linalg.norm(embedding)

            return embedding.tolist()
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return None

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

class DINOv2Embedder:
    def __init__(self, model_name="facebook/dinov2-base"):
        """
        Initialize the DINOv2 model for creating image embeddings.

        Args:
            model_name (str): Model name from Hugging Face model hub
                              Options: "facebook/dinov2-base", "facebook/dinov2-small",
                              "facebook/dinov2-large", "facebook/dinov2-giant"
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def embed_image(self, image_path):
        """
        Create embedding for a single image.

        Args:
            image_path (str): Path to the image file

        Returns:
            numpy.ndarray: Image embedding vector
        """
        try:
            image = Image.open(image_path).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            # Get [CLS] token embedding
            embedding = outputs.last_hidden_state[:, 0].cpu().numpy()
            return embedding[0]
        except Exception as e:
            print(f"Error embedding image {image_path}: {str(e)}")
            return None

    def embed_batch(self, image_paths, batch_size=16):
        """
        Create embeddings for a batch of images.

        Args:
            image_paths (list): List of image file paths
            batch_size (int): Number of images to process at once

        Returns:
            dict: Dictionary mapping image paths to their embeddings
        """
        embeddings = {}

        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(len(image_paths) + batch_size - 1)//batch_size}")

            valid_images = []
            valid_paths = []

            for path in batch_paths:
                try:
                    image = Image.open(path).convert("RGB")
                    valid_images.append(image)
                    valid_paths.append(path)
                except Exception as e:
                    print(f"Error opening image {path}: {str(e)}")

            if not valid_images:
                continue

            inputs = self.processor(images=valid_images, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            # Get [CLS] token embeddings
            batch_embeddings = outputs.last_hidden_state[:, 0].cpu().numpy()

            for idx, path in enumerate(valid_paths):
                embeddings[path] = batch_embeddings[idx]

        return embeddings

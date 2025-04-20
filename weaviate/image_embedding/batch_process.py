#!/usr/bin/env python
import argparse
import os
from pathlib import Path

import weaviate
from dinov2_embedder import DINOv2Embedder
from PIL import Image
from tqdm import tqdm


def get_image_metadata(image_path: Path) -> dict:
    """Extract width, height, format, and size_kb for an image."""
    try:
        with Image.open(image_path) as img:
            return {
                "width": img.width,
                "height": img.height,
                "format": img.format,
                "size_kb": os.path.getsize(image_path) / 1024,
            }
    except Exception as e:
        print(f"⚠️ Metadata error for {image_path}: {e}")
        return {}


def ensure_collection_exists(
    client: weaviate.WeaviateClient, embedding_dim: int
) -> None:
    """
    Ensure that an 'Image' collection exists; if not, create it with:
      - no vectorizer (vectors provided by us),
      - hnsw index with optimized parameters,
      - filename, path, metadata properties.
    """
    # List all existing collections
    existing = client.collections.list_all()
    if "Image" not in existing:
        class_schema = {
            "class": "Image",
            "vectorizer": "none",
            "vectorIndexType": "hnsw",
            "vectorIndexConfig": {
                "ef": 200,
                "efConstruction": 128,
                "maxConnections": 16,
                "vectorCacheMaxObjects": 1000000,
            },
            "properties": [
                {
                    "name": "filename",
                    "dataType": ["text"],  # list, not set
                },
                {
                    "name": "path",
                    "dataType": ["text"],  # list, not set
                },
                {
                    "name": "metadata",
                    "dataType": ["object"],  # list, not set
                    "nestedProperties": [  # list, not set
                        {"name": "width", "dataType": ["int"]},
                        {"name": "height", "dataType": ["int"]},
                        {"name": "format", "dataType": ["text"]},
                        {"name": "size_kb", "dataType": ["number"]},
                    ],
                },
            ],
        }
        # Create via v3‑style JSON (supports hnsw) :contentReference[oaicite:5]{index=5}
        client.collections.create_from_dict(class_schema)
        print(f"✅ Created 'Image' collection (dim={embedding_dim})")
    else:
        print("ℹ️ Collection 'Image' already exists")


def batch_process_images(
    directory_path: str,
    client: weaviate.WeaviateClient,
    embedder,
    batch_size: int = 32,
) -> None:
    """Scan for images, embed in batches, and upload to Weaviate."""
    # Collect image file paths
    p = Path(directory_path)
    supported = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
    files = [f for f in p.rglob("*") if f.suffix.lower() in supported]
    print(f"Found {len(files)} images")

    # Ensure the collection exists
    ensure_collection_exists(client, embedder.get_embedding_dimension())

    # Insert in batches
    for i in range(0, len(files), batch_size):
        batch_files = files[i : i + batch_size]
        objs = []
        print(f"→ Batch {i//batch_size+1}/{(len(files)-1)//batch_size+1}")
        for path in tqdm(batch_files):
            vec = embedder.get_embedding(path)
            if vec is None:
                continue
            objs.append(
                {
                    "properties": {
                        "filename": path.name,
                        "path": str(path.relative_to(p)),
                        "metadata": get_image_metadata(path),
                    },
                    "vector": vec,
                }
            )

        # Process objects in batch with Weaviate v4 API
        image_collection = client.collections.get("Image")
        # Add objects one by one (no context manager in v4)
        for o in objs:
            image_collection.data.insert(properties=o["properties"], vector=o["vector"])
        print(f"✔️ Inserted {len(objs)} objects")


def main():
    parser = argparse.ArgumentParser(
        description="Batch import DINOv2 embeddings into Weaviate v4"
    )
    parser.add_argument("directory", help="Directory containing images")
    parser.add_argument(
        "--model_size",
        choices=["small", "base", "large", "giant"],
        default="base",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--weaviate_url",
        default="http://localhost:8080",
        help="HTTP URL of your Weaviate instance",
    )
    args = parser.parse_args()

    # Initialize embedder (device selection printed internally)
    embedder = DINOv2Embedder(model_size=args.model_size)
    print(f"Using device: {embedder.device}")

    # Instantiate v4 client (synchronous, default) :contentReference[oaicite:7]{index=7}
    client = weaviate.WeaviateClient(
        connection_params=weaviate.connect.ConnectionParams.from_url(
            url=args.weaviate_url, grpc_port=50051
        ),
        skip_init_checks=True,
    )
    client.connect()

    try:
        batch_process_images(args.directory, client, embedder, args.batch_size)
    finally:
        client.close()


if __name__ == "__main__":
    main()

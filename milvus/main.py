import os
import argparse
from dinov2_embedder import DINOv2Embedder
from milvus_setup import MilvusImageDB

def get_image_paths(directory, extensions=('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
    """Get all image paths from a directory"""
    image_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(extensions):
                image_paths.append(os.path.join(root, file))
    return image_paths

def embed_and_insert_images(directory, embedder, db):
    """Embed all images in the directory and insert into Milvus"""
    image_paths = get_image_paths(directory)
    print(f"Found {len(image_paths)} images")

    # Create embeddings for all images
    embeddings_dict = embedder.embed_batch(image_paths)
    print(f"Created embeddings for {len(embeddings_dict)} images")

    # Insert the embeddings into Milvus
    db.insert_embeddings(embeddings_dict)

def parse_args():
    parser = argparse.ArgumentParser(description="Image Similarity Search with Milvus and DINOv2")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Index command
    index_parser = subparsers.add_parser("index", help="Index images into Milvus")
    index_parser.add_argument("--directory", "-d", required=True, help="Directory containing images to index")
    index_parser.add_argument("--model", "-m", default="facebook/dinov2-base", help="DINOv2 model variant")

    # Search command
    search_parser = subparsers.add_parser("search", help="Search for similar images")
    search_parser.add_argument("--query", "-q", required=True, help="Path to query image")
    search_parser.add_argument("--top_k", "-k", type=int, default=3, help="Number of similar images to return")
    search_parser.add_argument("--model", "-m", default="facebook/dinov2-base", help="DINOv2 model variant")

    return parser.parse_args()

def main():
    args = parse_args()

    # Initialize Milvus
    db = MilvusImageDB()

    if args.command == "index":
        print(f"Indexing images from {args.directory}")
        embedder = DINOv2Embedder(model_name=args.model)
        embed_and_insert_images(args.directory, embedder, db)
        print("Indexing complete")

    elif args.command == "search":
        print(f"Searching for images similar to {args.query}")
        embedder = DINOv2Embedder(model_name=args.model)

        # Embed query image
        query_embedding = embedder.embed_image(args.query)

        if query_embedding is None:
            print(f"Failed to embed query image {args.query}")
            return

        # Search for similar images
        results = db.search(query_embedding, top_k=args.top_k)

        print(f"\nFound {len(results)} similar images:")
        for i, result in enumerate(results):
            print(f"{i+1}. {result['image_path']} (distance: {result['distance']:.4f})")

    db.close()

if __name__ == "__main__":
    main()

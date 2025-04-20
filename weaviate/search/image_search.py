import argparse
import os
import sys
from pathlib import Path

import weaviate

# Add parent directory to sys.path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from image_embedding.dinov2_embedder import DINOv2Embedder


def image_to_image_search(client, embedder, query_image_path, limit=5):
    """Find similar images to a query image"""
    # Generate embedding for query image
    query_embedding = embedder.get_embedding(query_image_path)

    if query_embedding is None:
        print(f"Error: Could not generate embedding for {query_image_path}")
        return []

    # Get the Image collection
    image_collection = client.collections.get("Image")

    # Search in Weaviate
    try:
        print("Trying search with simplified properties...")
        response = image_collection.query.near_vector(
            near_vector=query_embedding,
            limit=limit,
            return_properties=["filename", "path"],
        )

        # Process results with minimal metadata
        image_results = []
        for obj in response.objects:
            image_results.append(
                {
                    "filename": obj.properties["filename"],
                    "path": obj.properties["path"],
                    "metadata": {},  # Empty metadata to avoid serialization issues
                    "similarity": obj.metadata.certainty,
                }
            )

        print("Search successful with simplified properties")
        return image_results

    except Exception as e:
        print(f"Error during search: {e}")
        print("Try restarting the Weaviate container if this issue persists")
        return []


def print_search_results(results, query_image):
    """Print search results in a readable format"""
    if not results:
        print(f"No similar images found for {query_image}")
        return

    print(f"\nFound {len(results)} similar images to {query_image}:")
    print("-" * 50)
    for i, result in enumerate(results):
        print(f"{i+1}. {result['filename']}")
        print(f"   Path: {result['path']}")

        # Safely print similarity score if available
        similarity = result.get("similarity")
        if similarity is not None:
            print(f"   Similarity: {similarity:.4f}")
        else:
            print("   Similarity: Not available")

        # Safely print metadata if available
        metadata = result.get("metadata", {})
        if metadata:
            # Check if we have dimension information
            if "width" in metadata and "height" in metadata:
                print(f"   Dimensions: {metadata['width']}x{metadata['height']}")

            # Check if we have format information
            if "format" in metadata:
                print(f"   Format: {metadata['format']}")

            # Check if we have size information
            if "size_kb" in metadata:
                print(f"   Size: {metadata['size_kb']:.2f} KB")
        else:
            print("   Metadata: Not available")

        print("-" * 50)


def main():
    parser = argparse.ArgumentParser(description="Image similarity search with DINOv2")
    parser.add_argument("query_image", help="Path to query image")
    parser.add_argument(
        "--model_size",
        choices=["small", "base", "large", "giant"],
        default="base",
        help="DINOv2 model size",
    )
    parser.add_argument(
        "--limit", type=int, default=5, help="Number of results to return"
    )
    parser.add_argument(
        "--weaviate_url", default="http://localhost:8080", help="Weaviate server URL"
    )

    args = parser.parse_args()

    # Initialize DINOv2 embedder
    embedder = DINOv2Embedder(model_size=args.model_size)
    print(f"Using device: {embedder.device}")

    # Connect to Weaviate using WeaviateClient with robust connection settings
    print(f"Trying to connect to Weaviate at {args.weaviate_url}")

    # Ensure Docker containers are running
    print("Tip: Make sure Weaviate container is running with proper port mapping:")
    print("  docker-compose -f docker/docker-compose.yml up -d")

    # Try multiple connection methods
    try:
        # Try with explicit gRPC settings
        connection_params = weaviate.connect.ConnectionParams.from_url(
            url=args.weaviate_url, grpc_port=50051
        )

        client = weaviate.WeaviateClient(
            connection_params=connection_params, skip_init_checks=True
        )

        client.connect()
        print("Successfully connected to Weaviate with gRPC")
    except Exception as e:
        print(f"All connection attempts failed: {e}")
        print("\n======== TROUBLESHOOTING TIPS ========")
        print("1. Ensure Weaviate container is running:")
        print("   docker ps | grep weaviate")
        print("2. Verify port mappings in docker-compose.yml:")
        print("   - HTTP port 8080 should be mapped")
        print("   - gRPC port 50051 should be mapped")
        print("3. Check if container is healthy:")
        print("   docker logs $(docker ps -q --filter name=weaviate)")
        print("4. Try restarting the container:")
        print("   docker-compose -f docker/docker-compose.yml restart")
        print("=======================================")
        sys.exit(1)

    # Search for similar images
    results = image_to_image_search(client, embedder, args.query_image, args.limit)

    # Print results
    print_search_results(results, args.query_image)

    client.close()


if __name__ == "__main__":
    main()

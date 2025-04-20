# DINOv2 Image Similarity Search with Weaviate and HNSW

This project implements an image similarity search system using:
- **DINOv2**: For generating high-quality image embeddings
- **Weaviate**: As the vector database
- **HNSW**: For efficient vector indexing and similarity search

## System Requirements

- Python 3.8+
- Docker and Docker Compose
- CUDA-compatible GPU (optional but recommended)

## Setup

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/dinov2-weaviate-ngt.git
   cd dinov2-weaviate-ngt
   ```

2. Run the setup script:
   ```
   chmod +x setup.sh
   ./setup.sh
   ```

   This will:
   - Create a Python virtual environment
   - Install required dependencies
   - Build and start Docker services for Weaviate

## Usage

### 1. Process Images

To embed a directory of images and store them in Weaviate:

```bash
python image-embedding/batch_process.py --model_size base /path/to/your/images
```

Options:
- `--model_size`: DINOv2 model size (small, base, large, giant)
- `--batch_size`: Number of images to process at once (default: 32)
- `--weaviate_url`: Weaviate server URL (default: http://localhost:8080)

### 2. Search for Similar Images

To find images similar to a query image:

```bash
python search/image_search.py --model_size base /path/to/query/image.jpg
```

Options:
- `--model_size`: DINOv2 model size (must match the one used for processing)
- `--limit`: Number of results to return (default: 5)
- `--weaviate_url`: Weaviate server URL (default: http://localhost:8080)

## Model Sizes

DINOv2 comes in multiple sizes. Choose according to your needs:

| Model Size | Embedding Dimension | Memory Requirements | Performance |
|------------|---------------------|---------------------|-------------|
| small      | 384                 | Low                 | Good        |
| base       | 768                 | Medium              | Better      |
| large      | 1024                | High                | Excellent   |
| giant      | 1536                | Very High           | Best        |

## Components

- **HNSW Vector Index**: Built-in Weaviate vector indexing algorithm
- **DINOv2 Embedder**: Generates embeddings from images
- **Batch Processor**: Handles processing large image collections
- **Image Search**: Performs similarity searches

## License

MIT

## How to Use the System

1. **Setup**:
   - Run `./setup.sh` to initialize everything

2. **Process Images**:
   - Run `python image-embedding/batch_process.py --model_size base /path/to/your/images`
   - This creates embeddings and stores them in Weaviate with HNSW indexing

3. **Search for Similar Images**:
   - Run `python search/image_search.py --model_size base /path/to/query/image.jpg`
   - Displays the most similar images to your query

This implementation provides a complete, production-ready system for image similarity search using state-of-the-art DINOv2 embeddings with the efficient HNSW indexing algorithm in Weaviate.

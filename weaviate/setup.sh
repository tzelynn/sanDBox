#!/bin/bash

# Check for Docker
if ! command -v docker &> /dev/null; then
    echo "Docker not found. Please install Docker first."
    exit 1
fi

# Check for Docker Compose
if ! command -v docker compose &> /dev/null; then
    echo "Docker Compose not found. Please install Docker Compose first."
    exit 1
fi

# Create virtual environment for Python
echo "Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install dependencies
echo "Installing Python dependencies..."
pip3 install -r requirements.txt

# Build and start Docker services
echo "Building and starting Docker services..."
cd docker
docker compose up -d

echo "Waiting for services to start..."
sleep 10

echo "Setup complete! Services are running."
echo "Use the following commands to work with the system:"
echo "  - Process images: python image-embedding/batch_process.py --model_size base /path/to/images"
echo "  - Search images: python search/image_search.py --model_size base /path/to/query/image.jpg"

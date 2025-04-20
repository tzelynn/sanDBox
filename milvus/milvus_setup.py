# milvus_setup.py
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility

class MilvusImageDB:
    def __init__(self, collection_name="image_collection", host="localhost", port="19530"):
        """
        Initialize connection to Milvus and create collection if it doesn't exist.

        Args:
            collection_name (str): Name of the Milvus collection
            host (str): Milvus server host
            port (str): Milvus server port
        """
        self.collection_name = collection_name

        # Connect to Milvus
        connections.connect(host=host, port=port)

        # Check if collection exists, if not create it
        if not utility.has_collection(self.collection_name):
            self._create_collection()

        self.collection = Collection(self.collection_name)

    def _create_collection(self):
        """Create a new collection with the appropriate schema"""
        # Define fields for the collection
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="image_path", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768)  # DINOv2-base dim=768
        ]

        # Create collection schema
        schema = CollectionSchema(fields=fields, description="Image collection for similarity search")

        # Create collection
        collection = Collection(name=self.collection_name, schema=schema)

        # Create HNSW index on the embedding field
        index_params = {
            "metric_type": "COSINE",  # or "IP" for inner product
            "index_type": "HNSW",
            "params": {
                "M": 16,  # Number of edges per node (higher = more accuracy but more memory)
                "efConstruction": 500  # Higher values build more accurate indices but take longer
            }
        }

        collection.create_index("embedding", index_params)
        print(f"Created collection '{self.collection_name}' with HNSW index")
        return collection

    def insert_embeddings(self, embeddings_dict):
        """
        Insert embeddings into Milvus.

        Args:
            embeddings_dict (dict): Dictionary mapping image paths to their embeddings
        """
        if not embeddings_dict:
            print("No embeddings to insert")
            return

        # Prepare data for insertion
        image_paths = list(embeddings_dict.keys())
        embedding_vectors = list(embeddings_dict.values())

        # Insert the data
        entities = [
            image_paths,
            embedding_vectors
        ]

        self.collection.insert(entities)
        self.collection.flush()
        print(f"Inserted {len(image_paths)} embeddings into collection")

    def load_collection(self):
        """Load collection into memory for searching"""
        try:
            self.collection.load()
            print(f"Collection '{self.collection_name}' loaded for searching")
        except Exception as e:
            print(f"Warning: {str(e)}")

    def search(self, query_embedding, top_k=5):
        """
        Search for similar images.

        Args:
            query_embedding (numpy.ndarray): Embedding vector of the query image
            top_k (int): Number of similar images to return

        Returns:
            list: List of dictionaries containing results
        """
        # Always load the collection before searching
        # This is safe to call multiple times - it's idempotent
        self.load_collection()

        search_params = {
            "metric_type": "COSINE",
            "params": {"ef": 100}  # Higher ef means more accurate search but slower
        }

        results = self.collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["image_path"]
        )

        # Format results
        formatted_results = []
        for hits in results:
            for hit in hits:
                formatted_results.append({
                    "image_path": hit.entity.get("image_path"),
                    "distance": hit.distance
                })

        return formatted_results

    def close(self):
        """Release collection and disconnect from Milvus"""
        try:
            self.collection.release()
        except Exception as e:
            print(f"Warning when releasing collection: {str(e)}")
        connections.disconnect("default")

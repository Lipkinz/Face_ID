import os
import numpy as np

def generate_dummy_embeddings(embeddings_dir, num_embeddings=5, embedding_dim=128):
    if not os.path.exists(embeddings_dir):
        os.makedirs(embeddings_dir)

    for i in range(1, num_embeddings + 1):
        identity = f"person{i}"
        dummy_embedding = np.random.rand(embedding_dim)
        np.save(os.path.join(embeddings_dir, f"{identity}.npy"), dummy_embedding)
        print(f"Saved embedding for {identity}")

if __name__ == "__main__":
    embeddings_dir = "embeddings"
    generate_dummy_embeddings(embeddings_dir)

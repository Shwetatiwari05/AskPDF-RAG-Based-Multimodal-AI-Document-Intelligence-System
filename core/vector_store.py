from core.embedder import get_model
import faiss
import numpy as np
import pickle


def create_vector_store(embeddings):

    # Embedding dimension
    dimension = embeddings.shape[1]

    # Cosine similarity search
    # Use Inner Product for cosine similarity since we normalized embeddings to unit length. 
    # This allows us to directly use the inner product as a measure of similarity without needing to compute cosine similarity separately.
    index = faiss.IndexFlatIP(dimension)  

    # Add embeddings to FAISS
    index.add(embeddings)   

    print(f"Total vectors stored: {index.ntotal}")

    return index

# Save and load vector store functions
def save_vector_store(     # Save both FAISS index and chunk metadata for later retrieval and search
    index,
    chunks,
    index_path="vector_store.index",
    chunks_path="chunks.pkl"
):

    # Save FAISS index
    faiss.write_index(index, index_path)  # Write the FAISS index to disk for later use in search operations

    # Save chunk metadata
    with open(chunks_path, "wb") as f:   # Write the chunk metadata (like chunk_id and text) to a pickle file for later retrieval when we get search results from the FAISS index
        pickle.dump(chunks, f)

    print("Vector store saved!")

# Load vector store function to load both the FAISS index and the chunk metadata for performing search operations later
def load_vector_store(
    index_path="vector_store.index",
    chunks_path="chunks.pkl"
):

    # Load FAISS index
    index = faiss.read_index(index_path)  # Read the FAISS index from disk to be able to perform search operations on it later when we have a query embedding to compare against the stored embeddings

    # Load chunk metadata
    with open(chunks_path, "rb") as f:  # Read the chunk metadata from the pickle file to be able to retrieve the original text chunks and their IDs when we get search results from the FAISS index based on the query embedding similarity
        chunks = pickle.load(f)   # Load the chunk metadata into memory so we can use it to map search results back to the original text chunks for displaying search results to the user

    print("Vector store loaded!")

    return index, chunks

# Perform semantic search by generating an embedding for the query and finding the most similar chunks in the vector store based on cosine similarity of embeddings
def search(query, index, chunks, top_k=5): 

    # Generate query embedding
    model = get_model()

    query_embedding = model.encode(
        [f"Represent this sentence for searching relevant passages: {query}"],
        normalize_embeddings=True

    )
    # Convert to float32
    query_embedding = np.array(
        query_embedding,
        dtype=np.float32
    )

    # Search similar vectors
    distances, indices = index.search(
        query_embedding,
        top_k     # Number of top similar chunks to retrieve based on cosine similarity of embeddings
    )

    results = []

    for i, idx in enumerate(indices[0]):

        # Safety check
        if idx == -1: # there are no more results to return (less than top_k results in the index), break out of the loop to avoid trying to access invalid indices in the chunks list.
            continue

        results.append({
            "chunk_id": chunks[idx]["chunk_id"],
            "text": chunks[idx]["text"],
            "score": float(distances[0][i])
        })

    return results


_reranker = None


def _get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    return _reranker


def rerank(query, results, top_k=3):
    # Create pairs of query and chunk text for reranking
    pairs = [[query, result["text"]] for result in results]

    # Get relevance scores from the reranker model
    reranker = _get_reranker()
    scores = reranker.predict(pairs)

    # Add reranking scores to results
    for i, result in enumerate(results):
        result["rerank_score"] = float(scores[i])

    # Sort results by reranking score in descending order and return the top_k results
    reranked = sorted(results, key=lambda x: x["rerank_score"], reverse=True)

    return reranked[:top_k]

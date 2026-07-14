from summarizer import client  # reuse the Gemini client we already set up
def chunk_text(text: str, chunk_size: int = 300, overlap: int = 60) -> list[str]:
    """
    Splits text into overlapping chunks, breaking only at word boundaries.
    chunk_size and overlap are approximate character counts.
    """
    words = text.split()
    chunks = []
    i = 0

    while i < len(words):
        current_chunk = []
        current_length = 0
        j = i

        # Build a chunk by adding words until we hit chunk_size
        while j < len(words) and current_length < chunk_size:
            current_chunk.append(words[j])
            current_length += len(words[j]) + 1
            j += 1

        chunks.append(" ".join(current_chunk))

        if j >= len(words):
            break

        # Figure out how many words to step back for overlap
        overlap_length = 0
        step_back = 0
        k = j - 1
        while k > i and overlap_length < overlap:
            overlap_length += len(words[k]) + 1
            step_back += 1
            k -= 1

        i = j - step_back  # move forward, but re-include the overlap words

    return [c for c in chunks if c]
from sentence_transformers import SentenceTransformer

# Load the embedding model once (this may take a moment the first time —
# it downloads the model, roughly 80MB)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_chunks(chunks: list[str]):
    """
    Converts a list of text chunks into embeddings (vectors of numbers).
    Returns a list of vectors, one per chunk, in the same order.
    """
    embeddings = embedding_model.encode(chunks)
    return embeddings

import chromadb

# Create a Chroma client and a collection (like a "table" for our chunks)
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="transcript_chunks")

def store_chunks(chunks: list[str]):
    """
    Embeds the chunks and stores them in the Chroma vector database,
    so they can be searched later by meaning.
    """
    embeddings = embed_chunks(chunks)

    # Chroma needs a unique string ID for each chunk
    ids = [f"chunk_{i}" for i in range(len(chunks))]

    collection.add(
        ids=ids,
        embeddings=embeddings.tolist(),  # Chroma wants plain lists, not numpy arrays
        documents=chunks
    )

    print(f"Stored {len(chunks)} chunks in the vector database")


def search_chunks(query: str, top_k: int = 2):
    """
    Given a question, finds the top_k most relevant chunks from the database.
    """
    query_embedding = embedding_model.encode([query])

    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=top_k
    )

    return results["documents"][0]  # the actual matching text chunks

def answer_question(query: str, top_k: int = 2) -> str:
    """
    Retrieves relevant chunks for the query, then asks Gemini to answer
    using only that retrieved context.
    """
    relevant_chunks = search_chunks(query, top_k=top_k)
    context = "\n\n".join(relevant_chunks)

    prompt = f"""You are answering a question based only on the context provided below.
If the answer is not in the context, say "I don't have enough information to answer that."
Do not make up information that isn't in the context.

Context:
{context}

Question: {query}

Answer:"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text

# Quick test
if __name__ == "__main__":
    sample_transcript = """
    Professor: Today we covered binary search trees. Remember, insertion is O(log n)
    on average but can degrade to O(n) if the tree becomes unbalanced. For next week,
    please read chapter 7 on AVL trees and submit the practice problems by Friday.
    Also, the mid-term will cover everything up to today's lecture. We also briefly
    touched on hash tables and how collision handling affects performance in the
    worst case scenario, which students should review before the exam as well.
    """

    chunks = chunk_text(sample_transcript, chunk_size=300, overlap=60)
    store_chunks(chunks)

    query = "What is due on Friday?"
    answer = answer_question(query, top_k=3)
    print(f"Query: {query}")
    print(f"Answer: {answer}")
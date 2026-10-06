# Load the search function and load_vector_store function to perform search operations on the vector store and retrieve the original text chunks based on search results from the FAISS index
# Load the embedding model to generate query embeddings for performing search operations on the vector store

from groq import Groq
from core.vector_store import search,rerank
from core.embedder import generate_embeddings
from dotenv import load_dotenv
import os
import re

# Load environment variables
load_dotenv()   

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ─── CONVERSATION MEMORY ──────────────────────────────────────────────────────
 
class ConversationMemory:
    """
    Stores last N exchanges (user question + assistant answer).
    Injected into prompt so LLM remembers context.
    """
    def __init__(self, max_exchanges: int = 10):
        self.max_exchanges = max_exchanges
        self.history = []   # list of {"role": "user"/"assistant", "content": "..."}
 
    def add(self, query: str, answer: str):
        self.history.append({"role": "user", "content": query})
        self.history.append({"role": "assistant", "content": answer})
        # Keep only last max_exchanges * 2 messages
        if len(self.history) > self.max_exchanges * 2:
            self.history = self.history[-(self.max_exchanges * 2):]
 
    def get_context_string(self) -> str:
        """Format history as readable string for prompt injection."""
        if not self.history:
            return ""
        lines = ["Previous conversation:"]
        for msg in self.history:
            role = "User" if msg["role"] == "user" else "Assistant"
            lines.append(f"{role}: {msg['content']}")
        return "\n".join(lines)
 
    def clear(self):
        self.history = []
        print("🧹 Conversation memory cleared.")

def generate_answer(query: str, chunks: list, index, memory: ConversationMemory = None) -> tuple:

    # Normalize query
    query = query.lower()
    query = query.replace("-", " ")
    query = re.sub(r'handcoded', 'hand coded', query)
    query = re.sub(r'\s+', ' ', query).strip()

    results = search(query, index, chunks, top_k=50)

    stopwords = {
    "what","is","are","the","a","an",
    "of","in","on","to","for","and",
    "explain","define","tell","me"
}
    important_terms = [
    word for word in query.split()
    if word not in stopwords
    ]
    
    for r in results:
        text_lower = r["text"].lower()
        keyword_hits = sum(
            1 for term in important_terms
            if term in text_lower
            )
        r["score"] += keyword_hits * 0.05

    results = rerank(query, results, top_k=10)

    print("\nAFTER RERANK")

    for i, r in enumerate(results):
        print(i+1, r["score"])
        print(r["text"][:300])

    """ Iterate through search results and build a context string that includes the text of the most relevant chunks to provide 
    as context for the LLM to generate an answer based on the document content. Each chunk's text is prefixed with its rank in the search results for clarity"""

    context=""
    for i,result in enumerate(results):  
        context+= f"\n[Chunk {i+1}]:\n{result['text'][:2000]}\n"  # Add the text of each relevant chunk to the context string, prefixed with its rank in the search results 
    
    # Build conversation history string
    history_str = memory.get_context_string() if memory else ""

    # Create a prompt for the LLM that includes instructions to only answer based on the provided context, and includes the relevant chunks retrieved from the vector store as context for answering the question.
    prompt = f"""
You are a document question-answering assistant.

Use ONLY the provided context.

Rules:
- Answer using the information available in the context.
- If the exact answer is not present but related information exists, provide the best possible explanation from the context.
- Combine information from multiple chunks whenever useful.
- For summaries, include all important points found in the context.
- Do NOT repeatedly say "the context does not define..." or "the document does not explicitly state...".
- Give the most useful answer possible from the retrieved text.
- Only return "Information not found in document" if the retrieved chunks are completely unrelated to the question.

{f"---{chr(10)}{history_str}{chr(10)}---" if history_str else ""}

Context:
{context}

Question: {query}

Answer:""" 

    response = client.chat.completions.create(
        model="llama-3.1-70b-versatile",
        temperature=0.2,
        max_tokens=700,
        messages=[{"role": "user", "content": prompt}]
    )
 
    answer = response.choices[0].message.content.strip()
 
    # Save to memory
    if memory:
        memory.add(query, answer)
 
    return answer, results

# ─── PDF SELECTOR ─────────────────────────────────────────────────────────────
 
def select_pdf() -> tuple:
    """
    Let user choose which ingested PDF to query.
    Returns (index, chunks, pdf_name)
    """
    from core.ingest import list_pdfs, load_pdf_store
 
    pdfs = list_pdfs()
 
    if not pdfs:
        print(" No PDFs ingested yet!")
        print("   Run: python -m ingest your_file.pdf")
        return None, None, None
 
    if len(pdfs) == 1:
        # Only one PDF — auto-select
        pdf = pdfs[0]
        print(f" Auto-selected: {pdf['pdf_name']}")
        index, chunks, meta = load_pdf_store(pdf["pdf_id"])
        return index, chunks, pdf["pdf_name"]
 
    # Multiple PDFs — let user choose
    print("\n" + "=" * 60)
    print("SELECT A PDF TO QUERY")
    print("=" * 60)
    for i, pdf in enumerate(pdfs, 1):
        print(f"  [{i}] {pdf['pdf_name']}  ({pdf['chunk_count']} chunks, {pdf['page_count']} pages)")
    print("=" * 60)
 
    while True:
        choice = input("Enter number: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(pdfs):
            pdf = pdfs[int(choice) - 1]
            index, chunks, meta = load_pdf_store(pdf["pdf_id"])
            print(f"\n Loaded: {pdf['pdf_name']}\n")
            return index, chunks, pdf["pdf_name"]
        print("Invalid choice — try again.")


if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("AI DOCUMENT INTELLIGENCE — RAG CHATBOT")
    print("=" * 60)
 
    # Select PDF
    index, chunks, pdf_name = select_pdf()
    if index is None:
        exit(1)
 
    # Initialize conversation memory
    memory = ConversationMemory(max_exchanges=12)
 
    print(f"\n Chatting with: {pdf_name}")
    print("Commands: [V] Voice  [T] Text  [C] Clear memory  [S] Switch PDF  [Q] Quit\n")
 
    while True:
        print("\n" + "─" * 40)
        print("[V] Voice  [T] Text  [C] Clear memory  [S] Switch PDF  [Q] Quit")
        choice = input("Choice: ").strip().lower()
 
        if choice == "q":
            print("Goodbye!")
            break
 
        elif choice == "c":
            memory.clear()
            continue
 
        elif choice == "s":
            # Switch to different PDF
            index, chunks, pdf_name = select_pdf()
            if index is not None:
                memory.clear()  # Clear memory when switching PDFs
                print(f"Now chatting with: {pdf_name}")
            continue
 
        elif choice == "v":
            try:
                from features.voice_handler import voice_to_query
                query = voice_to_query()
                if not query:
                    print("No speech detected — try again.")
                    continue
            except Exception as e:
                print(f"Voice error: {e}")
                query = input("Type your question instead: ").strip()
 
        else:
            query = input("Your question: ").strip()
 
        if not query:
            continue
 
        print("\n Searching...")
        answer, _results = generate_answer(query, chunks, index, memory)
        print(f"\n Answer:\n{answer}")
 
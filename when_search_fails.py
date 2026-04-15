from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util

def main():
    # 1. Prepare the text and chunks (from Topic 2)
    text = """Software licensing agreements typically contain several critical provisions that govern the relationship between the licensor and the licensee. The grant of license clause REF-2024-7B defines the scope of permitted use for Version 4.1.2 of the software, including whether the license is exclusive or non-exclusive, the geographic territory covered, and the specific editions included. Termination clauses outline the conditions under which either party may end the agreement, with common triggers including material breach, insolvency, or failure to pay royalties within the specified cure period of thirty days as defined under clause MSA-9942. Intellectual property ownership sections clarify that the licensor retains all rights to the underlying code, and that any modifications or derivative works created by the licensee during the term remain the property of the licensor unless explicitly stated otherwise in a written amendment signed by both parties. Liability limitations under error code LIC-ERR-403 cap the total damages recoverable by either party to the fees paid in the preceding twelve months, excluding cases of gross negligence or willful misconduct. Confidentiality obligations require both parties to protect non-public technical documentation, pricing structures, and customer data using at minimum the same degree of care applied to their own proprietary information."""
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300, 
        chunk_overlap=50, 
        separators=["\n\n", "\n", ". ", ", ", " ", ""]
    )
    chunks = text_splitter.split_text(text)
    
    # 2. Setup BM25 (Keyword Search)
    # BM25 requires tokenized text (a list of words for each chunk)
    # We do a very basic lowercase split for keyword matching
    tokenized_chunks = [chunk.lower().replace('.', '').replace(',', '').split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized_chunks)
    
    # 3. Setup Semantic Search (Embeddings)
    import logging
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    print("Loading Sentence Transformer model for Semantic Search...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    chunk_embeddings = model.encode(chunks, convert_to_tensor=True)
    
    # 4. Define 3 Queries designed to expose the strengths and weaknesses of both methods
    queries = [
        # Query 1: Vocabulary Mismatch (Concept mapping without exact keywords)
        "What is the maximum compensation if something goes wrong?",
        
        # Query 2: Exact Code/Keyword Search
        "MSA-9942",
        
        # Query 3: Semantic/Lexical Trap (Synonyms vs exact terms)
        "Who owns the rights to a spinoff app built using this product?"
    ]
    
    print("\n" + "="*60)
    print(" COMPARING BM25 (KEYWORD) VS SEMANTIC SEARCH")
    print("="*60)
    
    for i, query in enumerate(queries):
        print(f"\n--- Query {i+1}: '{query}' ---")
        
        # BM25 Search
        tokenized_query = query.lower().replace('?', '').split()
        bm25_scores = bm25.get_scores(tokenized_query)
        best_bm25_idx = bm25_scores.argmax()
        
        # Semantic Search
        query_embedding = model.encode(query, convert_to_tensor=True)
        semantic_scores = util.cos_sim(query_embedding, chunk_embeddings)[0]
        best_semantic_idx = semantic_scores.argmax().item()
        
        print(f"\n[BM25 Best Match] (Score: {bm25_scores[best_bm25_idx]:.4f})")
        # If score is 0, BM25 found absolutely nothing
        if bm25_scores[best_bm25_idx] == 0:
            print(" -> NO MATCH FOUND. (BM25 relies on exact word overlap)")
        else:
            print(f" -> {chunks[best_bm25_idx]}")
            
        print(f"\n[Semantic Best Match] (Score: {semantic_scores[best_semantic_idx]:.4f})")
        print(f" -> {chunks[best_semantic_idx]}")

if __name__ == "__main__":
    main()

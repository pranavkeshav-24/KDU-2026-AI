from langchain_text_splitters import RecursiveCharacterTextSplitter

def main():
    text = """Software licensing agreements typically contain several critical provisions that govern the relationship between the licensor and the licensee. The grant of license clause REF-2024-7B defines the scope of permitted use for Version 4.1.2 of the software, including whether the license is exclusive or non-exclusive, the geographic territory covered, and the specific editions included. Termination clauses outline the conditions under which either party may end the agreement, with common triggers including material breach, insolvency, or failure to pay royalties within the specified cure period of thirty days as defined under clause MSA-9942. Intellectual property ownership sections clarify that the licensor retains all rights to the underlying code, and that any modifications or derivative works created by the licensee during the term remain the property of the licensor unless explicitly stated otherwise in a written amendment signed by both parties. Liability limitations under error code LIC-ERR-403 cap the total damages recoverable by either party to the fees paid in the preceding twelve months, excluding cases of gross negligence or willful misconduct. Confidentiality obligations require both parties to protect non-public technical documentation, pricing structures, and customer data using at minimum the same degree of care applied to their own proprietary information."""

    # Defense of this strategy:
    # We use a relatively small chunk size (around 300) with a 50-character overlap.
    # The hierarchy of separators [". ", " ", ""] ensures we FIRST try to split by sentence.
    # If a sentence is wildly long (legal text often is), it falls back to splitting by spaces.
    # The overlap ensures that strict conditions (like "under clause MSA-9942") aren't fully severed
    # from the preceding context if a split happens close to it.
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", ", ", " ", ""]
    )

    chunks = text_splitter.split_text(text)

    print("--- RAG Chunking Results ---")
    print(f"Strategy: Recursive Character Chunking")
    print(f"Total chunks created: {len(chunks)}\n")
    
    for i, chunk in enumerate(chunks):
        print(f"--- Chunk {i+1} ({len(chunk)} chars) ---")
        print(chunk)
        print()

if __name__ == "__main__":
    main()

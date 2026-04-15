from sentence_transformers import SentenceTransformer, util

def main():
    # Load a lightweight, popular pre-trained sentence transformer model
    print("Loading model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # The Hypothesis:
    # We will use two sentences that share almost identical vocabulary and sentence structure,
    # but have completely opposite meanings. We expect the model might be "fooled" into 
    # giving a very high similarity score (> 0.80) because of the lexical overlap.
    sentence1 = "The startup's new product was a massive success, generating millions in revenue."
    sentence2 = "The startup's new product was a massive failure, losing millions in revenue."

    print("\n--- Sentences ---")
    print(f"1: {sentence1}")
    print(f"2: {sentence2}")

    # Compute sentence embeddings
    embeddings1 = model.encode(sentence1, convert_to_tensor=True)
    embeddings2 = model.encode(sentence2, convert_to_tensor=True)

    # Compute cosine similarity
    cosine_score = util.cos_sim(embeddings1, embeddings2)[0][0].item()

    print("\n--- Results ---")
    print(f"Cosine Similarity Score: {cosine_score:.4f}")

    if cosine_score > 0.8:
        print("\nConclusion: Hypothesis PROVEN! The model was fooled.")
        print("Even though the sentences mean the exact opposite, the heavy word overlap resulted in a very high similarity score.")
    else:
        print("\nConclusion: Hypothesis FAILED! The model is smart.")
        print("The model correctly distinguished the contrasting meanings despite the similar vocabulary.")

if __name__ == "__main__":
    main()
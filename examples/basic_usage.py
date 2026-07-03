"""
Basic Usage Example
===================
Demonstrates encoding texts and computing similarity using MiniEmbed.

This script shows the three core operations:
  1. Encoding raw text into dense vectors
  2. Computing pairwise similarity between two texts
  3. Building a full similarity matrix across sets of texts
"""

import sys
sys.path.insert(0, '..')

from src.inference import EmbeddingInference


def main():
    print("=" * 60)
    print("MiniEmbed - Basic Usage Example")
    print("=" * 60)
    
    # Load the model
    print("\nLoading model...")
    model = EmbeddingInference.from_pretrained("../models/mini")
    print("Model loaded.\n")
    
    # -------------------------------------------------------------------------
    # Example 1: Encode texts
    # -------------------------------------------------------------------------
    print("-" * 40)
    print("Example 1: Encoding Texts")
    print("-" * 40)
    
    texts = [
        "Machine learning is a branch of artificial intelligence",
        "Deep learning uses neural networks with many layers",
        "I love eating pizza on weekends"
    ]
    
    embeddings = model.encode(texts)
    print(f"Input: {len(texts)} texts")
    print(f"Output: {embeddings.shape}")  # (3, 256)
    
    # -------------------------------------------------------------------------
    # Example 2: Compute similarity
    # -------------------------------------------------------------------------
    print("\n" + "-" * 40)
    print("Example 2: Computing Similarity")
    print("-" * 40)
    
    pairs = [
        ("Machine learning is great", "AI is wonderful"),
        ("Machine learning is great", "I love pizza"),
        ("The cat sat on the mat", "A feline rested on the rug"),
    ]
    
    for text1, text2 in pairs:
        similarity = model.similarity(text1, text2)
        tag = "MATCH" if similarity > 0.5 else "  LOW"
        print(f"  [{tag}] {similarity:.4f} | '{text1}' vs '{text2}'")
    
    # -------------------------------------------------------------------------
    # Example 3: Pairwise similarity matrix
    # -------------------------------------------------------------------------
    print("\n" + "-" * 40)
    print("Example 3: Pairwise Similarity Matrix")
    print("-" * 40)
    
    texts_a = ["Machine learning", "Deep learning", "Natural language"]
    texts_b = ["AI models", "Neural networks", "Text processing"]
    
    similarity_matrix = model.pairwise_similarity(texts_a, texts_b)
    
    print("\nSimilarity Matrix:")
    print("              ", "  ".join(f"{t[:10]:>10}" for t in texts_b))
    for i, text in enumerate(texts_a):
        row = "  ".join(f"{similarity_matrix[i, j]:>10.4f}" for j in range(len(texts_b)))
        print(f"{text[:12]:>12}: {row}")
    
    print("\nDone.")


if __name__ == "__main__":
    main()

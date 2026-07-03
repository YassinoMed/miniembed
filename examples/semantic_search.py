"""
Semantic Search Example
=======================
Demonstrates how to use MiniEmbed for document retrieval.

The model encodes a query and a corpus of documents into the same vector space,
then ranks documents by cosine similarity to the query. This finds results based
on meaning, not keyword overlap.
"""

import sys
sys.path.insert(0, '..')

from src.inference import EmbeddingInference


def main():
    print("=" * 60)
    print("MiniEmbed - Semantic Search Example")
    print("=" * 60)
    
    # Load the model
    print("\nLoading model...")
    model = EmbeddingInference.from_pretrained("../models/mini")
    print("Model loaded.\n")
    
    # -------------------------------------------------------------------------
    # Document collection
    # -------------------------------------------------------------------------
    documents = [
        "Python is a high-level programming language known for its simplicity",
        "Machine learning algorithms can learn patterns from data",
        "The weather today is sunny with a high of 75 degrees",
        "Neural networks are computational models inspired by the brain",
        "JavaScript is widely used for web development",
        "Deep learning has revolutionized computer vision and NLP",
        "Cats are popular pets known for their independence",
        "TensorFlow and PyTorch are popular deep learning frameworks",
        "The stock market showed strong gains today",
        "Natural language processing helps computers understand text"
    ]
    
    print(f"Document Collection: {len(documents)} documents")
    for i, doc in enumerate(documents, 1):
        print(f"   {i}. {doc[:60]}...")
    
    # -------------------------------------------------------------------------
    # Search queries
    # -------------------------------------------------------------------------
    queries = [
        "How do AI systems learn from examples?",
        "What programming language is good for beginners?",
        "Tell me about artificial neural networks",
    ]
    
    print("\n" + "=" * 60)
    print("Search Results")
    print("=" * 60)
    
    for query in queries:
        print(f"\n  Query: \"{query}\"")
        print("-" * 50)
        
        results = model.search(query, documents, top_k=3)
        
        for r in results:
            score = r['score']
            if score > 0.6:
                tag = "[HIGH]"
            elif score > 0.4:
                tag = "[ MED]"
            else:
                tag = "[ LOW]"
            
            print(f"   {tag} #{r['rank']} (score: {score:.4f})")
            print(f"         {r['text']}")
    
    # -------------------------------------------------------------------------
    # Interactive search (optional)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Interactive Search")
    print("=" * 60)
    print("Enter your own queries (type 'quit' to exit):\n")
    
    while True:
        try:
            query = input("  Query: ").strip()
            if query.lower() in ['quit', 'exit', 'q']:
                break
            if not query:
                continue
            
            results = model.search(query, documents, top_k=3)
            
            print("\n   Results:")
            for r in results:
                print(f"   - [{r['score']:.3f}] {r['text'][:70]}...")
            print()
            
        except (KeyboardInterrupt, EOFError):
            break
    
    print("\nDone.")


if __name__ == "__main__":
    main()

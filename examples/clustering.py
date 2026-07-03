"""
Text Clustering Example
=======================
Demonstrates how to cluster texts by semantic similarity using MiniEmbed.

The model encodes each text into a dense vector. K-Means clustering then
groups these vectors by proximity in the embedding space, even if the texts
share no common words.
"""

import sys
sys.path.insert(0, '..')

from src.inference import EmbeddingInference


def main():
    print("=" * 60)
    print("MiniEmbed - Text Clustering Example")
    print("=" * 60)
    
    # Load the model
    print("\nLoading model...")
    model = EmbeddingInference.from_pretrained("../models/mini")
    print("Model loaded.\n")
    
    # -------------------------------------------------------------------------
    # Text collection (mixed topics)
    # -------------------------------------------------------------------------
    texts = [
        # Technology
        "Python is a versatile programming language",
        "Machine learning models learn from data",
        "JavaScript is used for web development",
        "Neural networks process information like the brain",
        "Software engineering involves designing systems",
        
        # Food
        "Pizza is my favorite Italian dish",
        "Sushi is a traditional Japanese cuisine",
        "Tacos are delicious Mexican street food",
        "Pasta with marinara sauce is comforting",
        "Ramen noodles are popular in Japan",
        
        # Sports
        "Football is the most popular sport worldwide",
        "Basketball requires teamwork and skill",
        "Tennis is an exciting individual sport",
        "Swimming is great for cardiovascular health",
        "Soccer World Cup attracts billions of viewers",
        
        # Nature
        "Mountains offer breathtaking scenic views",
        "Oceans cover most of the Earth's surface",
        "Forests are home to diverse wildlife",
        "Rivers provide fresh water to ecosystems",
        "Deserts have extreme temperature variations",
    ]
    
    print(f"Text Collection: {len(texts)} texts (4 topics)")
    
    # -------------------------------------------------------------------------
    # Cluster texts
    # -------------------------------------------------------------------------
    print("\nClustering texts into 4 groups...")
    
    result = model.cluster_texts(texts, n_clusters=4)
    
    # -------------------------------------------------------------------------
    # Display results
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Clustering Results")
    print("=" * 60)
    
    for cluster_id in sorted(result['texts_by_cluster'].keys()):
        cluster_texts = result['texts_by_cluster'][cluster_id]
        
        print(f"\n  Cluster {cluster_id + 1} ({len(cluster_texts)} texts)")
        print("-" * 40)
        
        for text in cluster_texts:
            print(f"   - {text}")
    
    # -------------------------------------------------------------------------
    # Evaluate clustering (simple check)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Clustering Analysis")
    print("=" * 60)
    
    # Expected groupings (approximate)
    expected = {
        "Technology": texts[0:5],
        "Food": texts[5:10],
        "Sports": texts[10:15],
        "Nature": texts[15:20],
    }
    
    print("\nLabels assigned to each text:")
    for i, (text, label) in enumerate(zip(texts, result['labels'])):
        topic = list(expected.keys())[i // 5]
        print(f"   [{label}] ({topic}) {text[:50]}...")
    
    print("\nDone.")


if __name__ == "__main__":
    main()

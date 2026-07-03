"""
Model Saving & Inference Module
===================================
Easy-to-use API for loading and running inference with the embedding model.
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
import os
from pathlib import Path
from typing import List, Dict, Union, Tuple

from .model import MiniTransformerEmbedding
from .tokenizer import SimpleTokenizer


class EmbeddingModelManager:
    """
    Handles saving and loading the embedding model.
    
    Save structure:
    model_dir/
    ├── config.json          # Model architecture config
    ├── model.pt             # Model weights
    ├── tokenizer.json       # Vocabulary
    └── training_info.json   # Training metadata (optional)
    """
    
    @staticmethod
    def save_model(
        model: MiniTransformerEmbedding,
        tokenizer: SimpleTokenizer,
        save_dir: str,
        training_info: dict = None
    ):
        """
        Save model, tokenizer, and config for later use.
        
        Args:
            model: Trained MiniTransformerEmbedding
            tokenizer: SimpleTokenizer with vocabulary
            save_dir: Directory to save model
            training_info: Optional training metadata
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Save model config
        config = {
            'vocab_size': len(tokenizer.word_to_id),
            'd_model': model.d_model,
            'num_heads': model.layers[0].attention.num_heads,
            'num_layers': len(model.layers),
            'd_ff': model.layers[0].feed_forward.linear1.out_features,
            'max_seq_len': model.positional_encoding.pe.size(1),
            'pad_token_id': model.pad_token_id,
            'size_name': save_dir.name # Use folder name as size name
        }
        
        with open(save_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        # 2. Save model weights
        torch.save(model.state_dict(), save_dir / 'model.pt')
        
        # 3. Save tokenizer vocabulary
        tokenizer.save(str(save_dir / 'tokenizer.json'))
        
        # 4. Save training info (optional)
        if training_info:
            with open(save_dir / 'training_info.json', 'w') as f:
                json.dump(training_info, f, indent=2)
        
        print(f"Model saved to: {save_dir}")
    
    @staticmethod
    def load_model(model_dir: str, device: str = None) -> Tuple[MiniTransformerEmbedding, SimpleTokenizer]:
        """
        Load model and tokenizer from a local directory or HuggingFace repo.
        
        Args:
            model_dir: Local directory path OR HuggingFace repo ID 
                       (e.g., "surazbhandari/miniembed")
            device: Device to load model on ('cpu', 'cuda', 'mps')
            
        Returns:
            (model, tokenizer) tuple
        """
        # Auto-detect HuggingFace repo ID (contains "/" but is not a local path)
        if '/' in model_dir and not os.path.exists(model_dir):
            model_dir = EmbeddingModelManager._download_from_hub(model_dir)
        
        model_dir = Path(model_dir)
        
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            elif torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        
        # 1. Load config
        config_path = model_dir / 'config.json'
                
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # 2. Load tokenizer
        tokenizer_path = model_dir / 'tokenizer.json'

        tokenizer = SimpleTokenizer(vocab_size=config['vocab_size'])
        tokenizer.load(str(tokenizer_path))
        
        # 3. Create and load model
        model = MiniTransformerEmbedding(
            vocab_size=config['vocab_size'],
            d_model=config['d_model'],
            num_heads=config['num_heads'],
            num_layers=config['num_layers'],
            d_ff=config['d_ff'],
            max_seq_len=config['max_seq_len'],
            pad_token_id=config['pad_token_id']
        )
        
        # Load weights (prefer safetensors)
        st_path = model_dir / 'model.safetensors'
        pt_path = model_dir / 'model.pt'
        
        if st_path.exists():
            from safetensors.torch import load_file
            state_dict = load_file(str(st_path), device=device)
        elif pt_path.exists():
            state_dict = torch.load(pt_path, map_location=device, weights_only=True)
        else:
            raise FileNotFoundError(f"Neither model.safetensors nor model.pt found in {model_dir}")
            
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        
        return model, tokenizer
    
    @staticmethod
    def _download_from_hub(repo_id: str) -> str:
        """
        Download model files from a HuggingFace repository.
        
        Args:
            repo_id: HuggingFace repo ID (e.g., "surazbhandari/miniembed")
            
        Returns:
            Local directory path containing the downloaded files.
        """
        try:
            from huggingface_hub import hf_hub_download, snapshot_download
        except ImportError:
            raise ImportError(
                "huggingface_hub is required to download models from HuggingFace. "
                "Install it with: pip install huggingface_hub"
            )
        
        # Download the full model snapshot
        local_dir = snapshot_download(
            repo_id=repo_id,
            allow_patterns=["config.json", "model.safetensors", "model.pt", "tokenizer.json", "training_info.json"],
        )
        
        return local_dir
    
    @staticmethod
    def list_models(base_dir: str = "models") -> List[str]:
        """
        List available model names in the base directory.
        
        Returns:
            List of directory names containing valid models
        """
        path = Path(base_dir)
        if not path.exists():
            return []
        return sorted([d.name for d in path.iterdir() if d.is_dir() and (d / "model.pt").exists()])

class EmbeddingInference:
    """
    High-level inference API for the embedding model.
    
    Usage:
        # From local directory
        model = EmbeddingInference.from_pretrained("./models/mini")
        
        # From HuggingFace
        model = EmbeddingInference.from_pretrained("surazbhandari/miniembed")
        
        # Encode texts
        embeddings = model.encode(["Hello world", "Machine learning"])
        
        # Compute similarity
        score = model.similarity("query", "document")
        
        # Semantic search
        results = model.search("python programming", documents)
    """
    
    def __init__(
        self, 
        model: MiniTransformerEmbedding, 
        tokenizer: SimpleTokenizer,
        device: str = 'cpu',
        max_length: int = 64
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length
        self.model.eval()
    
    @classmethod
    def from_pretrained(cls, model_dir: str, device: str = None):
        """
        Load model from a local directory or HuggingFace repo ID.
        
        Args:
            model_dir: Local path (e.g., "models/mini") or 
                       HuggingFace repo ID (e.g., "surazbhandari/miniembed")
            device: Device to load on ('cpu', 'cuda', 'mps'). Auto-detected if None.
        """
        model, tokenizer = EmbeddingModelManager.load_model(model_dir, device)
        if device is None:
            device = next(model.parameters()).device.type
        return cls(model, tokenizer, device)
    
    def encode(
        self, 
        texts: Union[str, List[str]], 
        batch_size: int = 32,
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Encode texts to embeddings.
        
        Args:
            texts: Single text or list of texts
            batch_size: Batch size for encoding
            show_progress: Show progress bar
            
        Returns:
            numpy array of shape (n_texts, d_model)
        """
        if isinstance(texts, str):
            texts = [texts]
        
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize
            encodings = [
                self.tokenizer.encode(t, self.max_length) 
                for t in batch_texts
            ]
            
            input_ids = torch.stack([e['input_ids'] for e in encodings]).to(self.device)
            attention_mask = torch.stack([e['attention_mask'] for e in encodings]).to(self.device)
            
            # Encode
            with torch.no_grad():
                embeddings = self.model.encode(input_ids, attention_mask)
            
            all_embeddings.append(embeddings.cpu().numpy())
        
        return np.vstack(all_embeddings)
    
    def similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between two texts."""
        emb1 = self.encode(text1)
        emb2 = self.encode(text2)
        return float(np.dot(emb1[0], emb2[0]))
    
    def pairwise_similarity(self, texts1: List[str], texts2: List[str]) -> np.ndarray:
        """
        Compute pairwise similarity between two lists.
        
        Returns:
            Matrix of shape (len(texts1), len(texts2))
        """
        emb1 = self.encode(texts1)
        emb2 = self.encode(texts2)
        return np.dot(emb1, emb2.T)
    
    def search(
        self, 
        query: str, 
        documents: List[str], 
        top_k: int = 5
    ) -> List[Dict]:
        """
        Semantic search: Find most similar documents to query.
        
        Args:
            query: Search query
            documents: List of documents to search
            top_k: Number of results to return
            
        Returns:
            List of dicts with 'text', 'score', 'rank'
        """
        query_emb = self.encode(query)
        doc_embs = self.encode(documents)
        
        # Compute similarities
        scores = np.dot(doc_embs, query_emb.T).flatten()
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for rank, idx in enumerate(top_indices, 1):
            results.append({
                'rank': rank,
                'text': documents[idx],
                'score': float(scores[idx]),
                'index': int(idx)
            })
        
        return results
    
    def cluster_texts(self, texts: List[str], n_clusters: int = 5) -> Dict:
        """
        Cluster texts by embedding similarity.
        
        Returns:
            Dict with 'labels' and 'texts_by_cluster'
        """
        from sklearn.cluster import KMeans
        
        embeddings = self.encode(texts)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        
        return {
            'labels': labels.tolist(),
            'centroids': kmeans.cluster_centers_,
            'texts_by_cluster': {
                i: [texts[j] for j in range(len(texts)) if labels[j] == i]
                for i in range(n_clusters)
            }
        }

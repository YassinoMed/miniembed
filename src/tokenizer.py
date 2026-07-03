"""
Simple Word-Level Tokenizer
==============================
A basic tokenizer for demonstration purposes.
Converts text to token IDs with special tokens.
"""

import re
import json
from typing import Dict, List, Optional
from collections import Counter
from tqdm import tqdm


class SimpleTokenizer:
    """
    A simple word-level tokenizer with special tokens.
    
    Special Tokens:
    - [PAD]: Padding token (id=0)
    - [UNK]: Unknown token (id=1)
    - [CLS]: Classification token (id=2)
    - [SEP]: Separator token (id=3)
    """
    
    def __init__(self, vocab_size: int = 30000):
        self.vocab_size = vocab_size
        
        # Special tokens
        self.special_tokens = {
            '[PAD]': 0,
            '[UNK]': 1,
            '[CLS]': 2,
            '[SEP]': 3,
        }
        
        # Word to ID mapping
        self.word_to_id: Dict[str, int] = dict(self.special_tokens)
        self.id_to_word: Dict[int, str] = {v: k for k, v in self.special_tokens.items()}
        
        # Special token IDs
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.cls_token_id = 2
        self.sep_token_id = 3
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Split text into tokens (simple word-level tokenization).
        
        Args:
            text: Input text string
            
        Returns:
            List of tokens
        """
        # Lowercase and basic cleaning
        text = text.lower().strip()
        
        # Simple word tokenization with punctuation handling
        tokens = re.findall(r'\b\w+\b|[^\w\s]', text)
        
        return tokens
    
    def build_vocab(self, texts: List[str], min_freq: int = 2):
        """
        Build vocabulary from a list of texts.
        
        Args:
            texts: List of text strings
            min_freq: Minimum frequency for a word to be included
        """
        # Count word frequencies
        word_counts = Counter()
        
        for text in tqdm(texts, desc="Building vocabulary"):
            tokens = self._tokenize(text)
            word_counts.update(tokens)
        
        # Sort by frequency and take top vocab_size - special_tokens
        max_words = self.vocab_size - len(self.special_tokens)
        
        sorted_words = sorted(
            word_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Add words to vocabulary
        for word, count in sorted_words[:max_words]:
            if count >= min_freq and word not in self.word_to_id:
                idx = len(self.word_to_id)
                self.word_to_id[word] = idx
                self.id_to_word[idx] = word
        
        print(f"Vocabulary size: {len(self.word_to_id)}")
    
    def encode(self, text: str, max_length: int = 128) -> Dict:
        """
        Encode text to token IDs with attention mask.
        
        Args:
            text: Input text string
            max_length: Maximum sequence length
            
        Returns:
            Dictionary with 'input_ids' and 'attention_mask' tensors
        """
        import torch
        
        # Tokenize
        tokens = self._tokenize(text)
        
        # Convert to IDs (with CLS and SEP)
        token_ids = [self.cls_token_id]
        
        for token in tokens[:max_length - 2]:  # Reserve space for CLS and SEP
            token_id = self.word_to_id.get(token, self.unk_token_id)
            token_ids.append(token_id)
        
        token_ids.append(self.sep_token_id)
        
        # Create attention mask
        attention_mask = [1] * len(token_ids)
        
        # Pad to max_length
        padding_length = max_length - len(token_ids)
        token_ids.extend([self.pad_token_id] * padding_length)
        attention_mask.extend([0] * padding_length)
        
        return {
            'input_ids': torch.tensor(token_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long)
        }
    
    def decode(self, token_ids: List[int]) -> str:
        """
        Decode token IDs back to text.
        
        Args:
            token_ids: List of token IDs
            
        Returns:
            Decoded text string
        """
        tokens = []
        for idx in token_ids:
            if idx in [self.pad_token_id, self.cls_token_id, self.sep_token_id]:
                continue
            token = self.id_to_word.get(idx, '[UNK]')
            tokens.append(token)
        return ' '.join(tokens)
    
    def save(self, path: str):
        """Save tokenizer vocabulary to JSON file."""
        data = {
            'vocab_size': self.vocab_size,
            'word_to_id': self.word_to_id,
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self, path: str):
        """Load tokenizer vocabulary from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        self.vocab_size = data['vocab_size']
        self.word_to_id = data['word_to_id']
        self.id_to_word = {int(v): k for k, v in self.word_to_id.items()}
    
    def __len__(self) -> int:
        return len(self.word_to_id)

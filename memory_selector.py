"""
Memory Selector Module for Humanized Conversation with Personality

This module implements the BERT-based memory selection mechanism.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    BertModel, 
    BertTokenizer, 
    BertConfig,
    get_linear_schedule_with_warmup
)
from typing import List, Dict, Tuple, Optional
import numpy as np
import logging
from dataclasses import dataclass

from config import bert_config, training_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class MemorySelectionSample:
    """A single sample for memory selection training"""
    context: str
    persona: str
    memory: str
    label: int  # 1 if memory is relevant, 0 otherwise

# =============================================================================
# Memory Selection Dataset
# =============================================================================

class MemorySelectionDataset(Dataset):
    """
    Dataset for training the BERT memory selector
    
    Each sample contains:
    - context: The dialogue history
    - persona: The personality profile
    - memory: A candidate memory item
    - label: Whether this memory is relevant (1) or not (0)
    """
    
    def __init__(
        self, 
        samples: List[MemorySelectionSample],
        tokenizer: BertTokenizer,
        max_length: int = 512
    ):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        # Combine context and persona as query
        query = f"[CLS] {sample.persona} [SEP] {sample.context} [SEP]"
        
        # Memory as candidate
        memory = sample.memory
        
        # Tokenize query and memory together
        encoding = self.tokenizer(
            query,
            memory,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'token_type_ids': encoding['token_type_ids'].squeeze(0),
            'label': torch.tensor(sample.label, dtype=torch.float)
        }

# =============================================================================
# BERT Memory Selector Model
# =============================================================================

class BertMemorySelector(nn.Module):
    """BERT-based memory selection model"""
    
    def __init__(
        self, 
        model_name: str = "bert-base-uncased",
        hidden_size: int = 768,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        
        # Classification head for relevance prediction
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self, 
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for memory relevance prediction
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            token_type_ids: Token type IDs [batch_size, seq_len]
            
        Returns:
            Relevance scores [batch_size, 1]
        """
        # Get BERT outputs
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # Use [CLS] token representation
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        
        # Predict relevance score
        relevance_score = self.classifier(cls_output)
        
        return relevance_score
    
    def compute_similarity(
        self,
        context_encoding: torch.Tensor,
        memory_encodings: torch.Tensor
    ) -> torch.Tensor:
        """Compute similarity between context and memory encodings"""
        # Normalize encodings
        context_norm = F.normalize(context_encoding, p=2, dim=-1)
        memory_norm = F.normalize(memory_encodings, p=2, dim=-1)
        
        # Compute cosine similarity
        similarity = torch.matmul(context_norm, memory_norm.transpose(-2, -1))
        
        return similarity

# =============================================================================
# Memory Selector Wrapper
# =============================================================================

class MemorySelector:
    """
    High-level wrapper for memory selection
    
    Implements the P_η component objective function:
    P(y_k | C, P, M) ≈ P_θ(y_k | C, P, m_k) · P_η(m_k | C, P, M)
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained(bert_config.model_name)
        
        if model_path:
            self.model = self._load_model(model_path)
        else:
            self.model = BertMemorySelector(
                model_name=bert_config.model_name,
                hidden_size=bert_config.hidden_size
            )
        
        self.model.to(self.device)
        self.model.eval()
    
    def _load_model(self, model_path: str) -> BertMemorySelector:
        """Load trained model from checkpoint"""
        model = BertMemorySelector(
            model_name=bert_config.model_name,
            hidden_size=bert_config.hidden_size
        )
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        return model
    
    def select_memory(
        self,
        context: str,
        persona: str,
        memories: List[str],
        top_k: int = 1
    ) -> List[Tuple[str, float]]:
        """
        Select the most relevant memories for the given context and persona
        
        Args:
            context: The dialogue context
            persona: The persona description
            memories: List of candidate memories
            top_k: Number of top memories to return
            
        Returns:
            List of (memory, score) tuples sorted by relevance
        """
        if not memories:
            return []
        
        scores = []
        
        with torch.no_grad():
            for memory in memories:
                # Prepare input
                query = f"[CLS] {persona} [SEP] {context} [SEP]"
                
                encoding = self.tokenizer(
                    query,
                    memory,
                    max_length=bert_config.max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                
                # Move to device
                input_ids = encoding['input_ids'].to(self.device)
                attention_mask = encoding['attention_mask'].to(self.device)
                token_type_ids = encoding['token_type_ids'].to(self.device)
                
                # Get relevance score
                score = self.model(input_ids, attention_mask, token_type_ids)
                scores.append(score.item())
        
        # Sort by score and return top_k
        scored_memories = list(zip(memories, scores))
        scored_memories.sort(key=lambda x: x[1], reverse=True)
        
        return scored_memories[:top_k]
    
    def batch_select_memory(
        self,
        contexts: List[str],
        personas: List[str],
        memories_list: List[List[str]],
        top_k: int = 1
    ) -> List[List[Tuple[str, float]]]:
        """Batch version of select_memory"""
        results = []
        
        for context, persona, memories in zip(contexts, personas, memories_list):
            selected = self.select_memory(context, persona, memories, top_k)
            results.append(selected)
        
        return results

# =============================================================================
# Training Functions
# =============================================================================

class MemorySelectorTrainer:
    """Trainer for the BERT memory selector"""
    
    def __init__(
        self,
        model: BertMemorySelector,
        tokenizer: BertTokenizer,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
    
    def train(
        self,
        train_samples: List[MemorySelectionSample],
        val_samples: Optional[List[MemorySelectionSample]] = None,
        num_epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        warmup_steps: int = 500,
        save_path: Optional[str] = None
    ):
        """
        Train the memory selector
        
        Args:
            train_samples: Training samples
            val_samples: Validation samples
            num_epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            warmup_steps: Number of warmup steps
            save_path: Path to save the trained model
        """
        # Create datasets
        train_dataset = MemorySelectionDataset(
            train_samples, 
            self.tokenizer,
            max_length=bert_config.max_length
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        
        # Setup optimizer and scheduler
        optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=bert_config.weight_decay
        )
        
        total_steps = len(train_loader) * num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # Loss function
        criterion = nn.BCELoss()
        
        # Training loop
        self.model.train()
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            total_loss = 0
            num_batches = 0
            
            for batch in train_loader:
                # Move to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                token_type_ids = batch['token_type_ids'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward pass
                outputs = self.model(input_ids, attention_mask, token_type_ids)
                loss = criterion(outputs.squeeze(), labels)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches
            logger.info(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")
            
            # Validation
            if val_samples:
                val_loss = self._validate(val_samples, batch_size, criterion)
                logger.info(f"Validation Loss: {val_loss:.4f}")
                
                if val_loss < best_val_loss and save_path:
                    best_val_loss = val_loss
                    torch.save(self.model.state_dict(), save_path)
                    logger.info(f"Model saved to {save_path}")
        
        if save_path and not val_samples:
            torch.save(self.model.state_dict(), save_path)
            logger.info(f"Final model saved to {save_path}")
    
    def _validate(
        self,
        val_samples: List[MemorySelectionSample],
        batch_size: int,
        criterion: nn.Module
    ) -> float:
        """Validate the model"""
        val_dataset = MemorySelectionDataset(
            val_samples,
            self.tokenizer,
            max_length=bert_config.max_length
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False
        )
        
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                token_type_ids = batch['token_type_ids'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask, token_type_ids)
                loss = criterion(outputs.squeeze(), labels)
                
                total_loss += loss.item()
                num_batches += 1
        
        self.model.train()
        return total_loss / num_batches

# =============================================================================
# Simple Memory Selector (Fallback without training)
# =============================================================================

class SimpleMemorySelector:
    """
    Simple memory selector using sentence embeddings
    
    This is a fallback when BERT model is not trained
    """
    
    def __init__(self):
        try:
            from sentence_transformers import SentenceTransformer
            self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        except ImportError:
            logger.warning("sentence-transformers not installed, using basic matching")
            self.encoder = None
    
    def select_memory(
        self,
        context: str,
        memories: List[str],
        top_k: int = 1
    ) -> List[Tuple[str, float]]:
        """Select memories using cosine similarity"""
        
        if not memories:
            return []
        
        if self.encoder is None:
            return self._basic_select(context, memories, top_k)
        
        # Encode context and memories
        context_embedding = self.encoder.encode(context)
        memory_embeddings = self.encoder.encode(memories)
        
        # Compute cosine similarities
        similarities = np.dot(memory_embeddings, context_embedding) / (
            np.linalg.norm(memory_embeddings, axis=1) * np.linalg.norm(context_embedding)
        )
        
        # Sort and return top_k
        indices = np.argsort(similarities)[::-1][:top_k]
        
        return [(memories[i], similarities[i]) for i in indices]
    
    def _basic_select(
        self,
        context: str,
        memories: List[str],
        top_k: int
    ) -> List[Tuple[str, float]]:
        """Basic word overlap selection"""
        context_words = set(context.lower().split())
        
        scores = []
        for memory in memories:
            memory_words = set(memory.lower().split())
            overlap = len(context_words.intersection(memory_words))
            scores.append(overlap / max(len(memory_words), 1))
        
        scored = list(zip(memories, scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        
        return scored[:top_k]

# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    # Example usage
    print("Testing Memory Selector...")
    
    # Test with simple selector
    simple_selector = SimpleMemorySelector()
    
    context = "I love hiking in the mountains and exploring nature."
    memories = [
        "I enjoy outdoor activities like camping and hiking.",
        "I work as a software engineer in a tech company.",
        "I have a dog named Max who loves to run.",
        "Nature photography is one of my hobbies.",
        "I prefer staying indoors and reading books."
    ]
    
    selected = simple_selector.select_memory(context, memories, top_k=2)
    
    print(f"\nContext: {context}")
    print("\nSelected memories:")
    for memory, score in selected:
        print(f"  [{score:.3f}] {memory}")

"""
Training Script for Humanized Conversation with Personality

This script handles the complete training pipeline for both:
1. BERT memory selector (P_η)
2. Llama3 response generator (P_θ)

Uses pre-split datasets (train_dataset, validation_dataset).
"""

import os
import argparse
import logging
import json
import random
from typing import List, Dict, Optional
from datetime import datetime

import torch
import numpy as np
from tqdm import tqdm

from config import (
    training_config,
    llama_config,
    bert_config,
    DATA_DIR,
    MODEL_DIR,
    OUTPUT_DIR,
    BIG5_PERSONA_PATH,
    PERSONA_CHAT_PATH
)
from data_processing import (
    DataLoader,
    TrainingDatasetBuilder,
    Persona,
    Conversation
)
from memory_selector import (
    BertMemorySelector,
    MemorySelectorTrainer,
    MemorySelectionSample
)
from response_generator import (
    ResponseGeneratorTrainer,
    ConversationSample
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# Utility Functions
# =============================================================================

def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def ensure_dirs():
    """Ensure necessary directories exist"""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# Data Preparation (Modified for Pre-split Data)
# =============================================================================

def prepare_memory_selection_data(
    train_loader: DataLoader,
    val_loader: DataLoader
) -> tuple:
    """
    Prepare data for BERT memory selector training using separate loaders.
    """
    logger.info("Preparing memory selection training data...")
    
    # 1. Build Training Data
    builder_train = TrainingDatasetBuilder(train_loader)
    raw_train = builder_train.build_memory_selection_dataset()
    train_samples = [
        MemorySelectionSample(
            context=s['context'],
            persona="", 
            memory=s['memory'],
            label=s['label']
        )
        for s in raw_train
    ]
    random.shuffle(train_samples)

    # 2. Build Validation Data
    builder_val = TrainingDatasetBuilder(val_loader)
    raw_val = builder_val.build_memory_selection_dataset()
    val_samples = [
        MemorySelectionSample(
            context=s['context'],
            persona="", 
            memory=s['memory'],
            label=s['label']
        )
        for s in raw_val
    ]
    
    logger.info(f"Memory selection data: {len(train_samples)} train, {len(val_samples)} val")
    
    return train_samples, val_samples

def prepare_response_generation_data(
    train_loader: DataLoader,
    val_loader: DataLoader
) -> tuple:
    """Prepare PERSONA-CHAT data for 7,151 train / 894 val split"""
    logger.info("Preparing response generation training data...")
    
    def _create_samples(loader: DataLoader, is_train: bool):
        samples = []
        
        # Map conversations to personas more systematically
        conversations = loader.persona_chat_data
        personas = loader.personas
        
        # Distribute conversations across personas
        convs_per_persona = len(conversations) // len(personas) if personas else 0
        
        for i, persona in enumerate(personas):
            system_prompt = persona.to_system_prompt()
            
            # Assign conversations to this persona
            start_idx = i * convs_per_persona
            end_idx = start_idx + convs_per_persona
            persona_convs = conversations[start_idx:end_idx]
            
            for conv in persona_convs:
                messages = [
                    {"role": turn.role, "content": turn.content}
                    for turn in conv.turns
                ]
                
                if messages:
                    samples.append(ConversationSample(
                        system_prompt=system_prompt,
                        messages=messages,
                        persona_big5=persona.big5,
                        selected_memory=None  # Would be populated if using memory
                    ))
        
        return samples
    
    train_samples = _create_samples(train_loader, is_train=True)
    val_samples = _create_samples(val_loader, is_train=False)
    
    logger.info(f"Response generation data: {len(train_samples)} train, {len(val_samples)} val")
    #logger.info(f"Expected from report: ~7,151 train, ~894 val")
    
    return train_samples, val_samples

# =============================================================================
# Training Functions
# =============================================================================

def train_memory_selector(
    train_samples: List[MemorySelectionSample],
    val_samples: List[MemorySelectionSample],
    output_dir: str,
    num_epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5
):
    """Train the BERT memory selector"""
    logger.info("=" * 50)
    logger.info("Training BERT Memory Selector")
    logger.info("=" * 50)
    
    from transformers import BertTokenizer
    
    tokenizer = BertTokenizer.from_pretrained(bert_config.model_name)
    model = BertMemorySelector(
        model_name=bert_config.model_name,
        hidden_size=bert_config.hidden_size
    )
    
    trainer = MemorySelectorTrainer(model=model, tokenizer=tokenizer)
    
    save_path = os.path.join(output_dir, "bert_memory_selector.pt")
    
    trainer.train(
        train_samples=train_samples,
        val_samples=val_samples,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_steps=bert_config.warmup_steps,
        save_path=save_path
    )
    
    logger.info(f"Memory selector saved to {save_path}")
    return save_path

def train_response_generator(
    train_samples: List[ConversationSample],
    val_samples: List[ConversationSample],
    output_dir: str,
    num_epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-4
):
    """Train the Llama3 response generator with QLoRA"""
    logger.info("=" * 50)
    logger.info("Training Llama3 Response Generator (QLoRA)")
    logger.info("=" * 50)
    
    if not torch.cuda.is_available():
        logger.warning("CUDA not available. Training will be very slow on CPU.")
    else:
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    
    trainer = ResponseGeneratorTrainer(model_name=llama_config.model_name)
    
    trainer.train(
        train_samples=train_samples,
        val_samples=val_samples,
        output_dir=output_dir,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate
    )
    
    logger.info(f"Response generator saved to {output_dir}")
    return output_dir

# =============================================================================
# Main Training Pipeline
# =============================================================================

def run_training_pipeline(args):
    """Run the complete training pipeline"""
    
    logger.info("=" * 60)
    logger.info("Humanized Conversation - Training Pipeline")
    logger.info("=" * 60)
    
    set_seed(args.seed)
    ensure_dirs()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(OUTPUT_DIR, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    # Save config
    with open(os.path.join(run_dir, "config.json"), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    logger.info(f"Output directory: {run_dir}")
    
    # --- Load Data (Separately for Train and Val) ---
    logger.info("\n" + "=" * 40)
    logger.info("Loading Datasets")
    logger.info("=" * 40)
    
    # Load Training Data
    logger.info(f"Loading Training Set: {args.train_dataset_path}")
    train_loader = DataLoader()
    if os.path.exists(args.train_dataset_path):
        train_loader.load_big5_personas(args.train_dataset_path)
    else:
        logger.error(f"Train dataset not found at {args.train_dataset_path}")
        return
        
    if args.persona_chat_path:
        train_loader.load_persona_chat(args.persona_chat_path)

    # Load Validation Data
    logger.info(f"Loading Validation Set: {args.val_dataset_path}")
    val_loader = DataLoader()
    if os.path.exists(args.val_dataset_path):
        val_loader.load_big5_personas(args.val_dataset_path)
    else:
        logger.error(f"Validation dataset not found at {args.val_dataset_path}")
        return

    if args.persona_chat_path:
        val_loader.load_persona_chat(args.persona_chat_path)
    
    logger.info(f"Train Personas: {len(train_loader.personas)}")
    logger.info(f"Valid Personas: {len(val_loader.personas)}")
    
    # --- Training ---

    # Train memory selector
    if args.train_bert:
        logger.info("\n" + "=" * 40)
        logger.info("Training Memory Selector (BERT)")
        logger.info("=" * 40)
        
        train_mem, val_mem = prepare_memory_selection_data(
            train_loader,
            val_loader
        )
        
        bert_output = os.path.join(run_dir, "memory_selector")
        os.makedirs(bert_output, exist_ok=True)
        
        train_memory_selector(
            train_samples=train_mem,
            val_samples=val_mem,
            output_dir=bert_output,
            num_epochs=args.bert_epochs,
            batch_size=args.bert_batch_size,
            learning_rate=args.bert_lr
        )
    
    # Train response generator
    if args.train_llama:
        logger.info("\n" + "=" * 40)
        logger.info("Training Response Generator (Llama3)")
        logger.info("=" * 40)
        
        train_resp, val_resp = prepare_response_generation_data(
            train_loader,
            val_loader
        )
        
        llama_output = os.path.join(run_dir, "response_generator")
        os.makedirs(llama_output, exist_ok=True)
        
        train_response_generator(
            train_samples=train_resp,
            val_samples=val_resp,
            output_dir=llama_output,
            num_epochs=args.llama_epochs,
            batch_size=args.llama_batch_size,
            learning_rate=args.llama_lr
        )
    
    logger.info("\n" + "=" * 60)
    logger.info("Training Complete!")
    logger.info("=" * 60)
    logger.info(f"Results saved to: {run_dir}")
    
    return run_dir

# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Train Humanized Conversation with Personality Models'
    )
    
    # Updated Data Paths Arguments
    parser.add_argument(
        '--train-dataset-path',
        type=str,
        default='data/train_dataset.json',
        help='Path to training personas JSON file'
    )
    parser.add_argument(
        '--val-dataset-path',
        type=str,
        default='data/validation_dataset.json',
        help='Path to validation personas JSON file'
    )
    parser.add_argument(
        '--persona-chat-path',
        type=str,
        default=PERSONA_CHAT_PATH,
        help='Path to PERSONA-CHAT CSV file'
    )
    
    # Training options
    parser.add_argument(
        '--train-bert',
        action='store_true',
        help='Train BERT memory selector'
    )
    parser.add_argument(
        '--train-llama',
        action='store_true',
        help='Train Llama3 response generator'
    )
    parser.add_argument(
        '--train-all',
        action='store_true',
        help='Train all models'
    )
    
    # BERT hyperparameters
    parser.add_argument('--bert-epochs', type=int, default=3)
    parser.add_argument('--bert-batch-size', type=int, default=16)
    parser.add_argument('--bert-lr', type=float, default=2e-5)
    
    # Llama hyperparameters
    parser.add_argument('--llama-epochs', type=int, default=3)
    parser.add_argument('--llama-batch-size', type=int, default=4)
    parser.add_argument('--llama-lr', type=float, default=2e-4)
    
    # General options
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    # Handle --train-all
    if args.train_all:
        args.train_bert = True
        args.train_llama = True
    
    # Check if anything to train
    if not args.train_bert and not args.train_llama:
        logger.warning("No training specified. Use --train-bert, --train-llama, or --train-all")
        logger.info("Running in demo mode...")
        args.train_bert = True 
    
    # Run training
    run_training_pipeline(args)

if __name__ == '__main__':
    main()

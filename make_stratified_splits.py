#!/usr/bin/env python3
"""
Stratified Dataset Splitting for Big-5 Personas
"""

import json
import random
import os
from typing import Dict, List, Tuple
from collections import defaultdict

def stratified_split_personas(
    all_personas: List[Dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Split personas with balanced Big-5 representation across all 32 combinations.

    Args:
        all_personas: List of persona dictionaries with 'big-5' key
        train_ratio: Proportion for training set (default: 0.8)
        val_ratio: Proportion for validation set (default: 0.1)
        test_ratio: Proportion for test set (default: 0.1)
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_personas, val_personas, test_personas)
    """
    
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Group personas by Big-5 combination
    by_big5 = defaultdict(list)
    for persona in all_personas:
        big5_key = persona.get('big-5', '')
        if not big5_key:
            print(f"Warning: Persona {persona.get('index', '?')} missing 'big-5' key")
            continue
        by_big5[big5_key].append(persona)
    
    # Validate we have all 32 combinations
    num_combinations = len(by_big5)
    print(f"Found {num_combinations} unique Big-5 combinations")
    
    if num_combinations != 32:
        print(f"WARNING: Expected 32 combinations, got {num_combinations}")
        print("Missing combinations will result in unbalanced test set")
    
    # Display distribution
    print("\nPersonas per Big-5 combination:")
    combo_sizes = [len(personas) for personas in by_big5.values()]
    print(f"  Min: {min(combo_sizes)}, Max: {max(combo_sizes)}, Avg: {sum(combo_sizes)/len(combo_sizes):.1f}")
    
    # Initialize result sets
    train_set = []
    val_set = []
    test_set = []
    
    # Split each Big-5 combination proportionally
    for big5_combo, personas in sorted(by_big5.items()):
        n = len(personas)
        
        # Shuffle personas within this combination
        random.shuffle(personas)
        
        # Calculate split points
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        # n_test = n - n_train - n_val (remaining)
        
        # Split
        train_set.extend(personas[:n_train])
        val_set.extend(personas[n_train:n_train+n_val])
        test_set.extend(personas[n_train+n_val:])
        
        # Log per-combination split (useful for debugging)
        # print(f"  {big5_combo}: {n_train} train, {n_val} val, {n-n_train-n_val} test")
    
    print(f"\nFinal split:")
    print(f"  Train: {len(train_set)} personas")
    print(f"  Val:   {len(val_set)} personas")
    print(f"  Test:  {len(test_set)} personas")
    print(f"  Total: {len(train_set) + len(val_set) + len(test_set)} personas")
    
    # Verify no overlap
    train_indices = {p['index'] for p in train_set}
    val_indices = {p['index'] for p in val_set}
    test_indices = {p['index'] for p in test_set}
    
    assert len(train_indices & val_indices) == 0, "Train/Val overlap detected"
    assert len(train_indices & test_indices) == 0, "Train/Test overlap detected"
    assert len(val_indices & test_indices) == 0, "Val/Test overlap detected"
    
    print("✓ No overlaps detected")
    
    return train_set, val_set, test_set

def verify_stratification(train_set: List[Dict], val_set: List[Dict], test_set: List[Dict]):
    """
    Verify that all 32 Big-5 combinations are represented in each split.
    """
    print("\n" + "="*60)
    print("Stratification Verification")
    print("="*60)
    
    def count_combinations(dataset: List[Dict], name: str):
        combos = defaultdict(int)
        for p in dataset:
            combos[p['big-5']] += 1
        print(f"\n{name} Set:")
        print(f"  Unique combinations: {len(combos)}/32")
        if len(combos) < 32:
            missing = 32 - len(combos)
            print(f"WARNING: {missing} combinations missing!")
        else:
            print(f"All combinations represented")
        return combos
    
    train_combos = count_combinations(train_set, "Train")
    val_combos = count_combinations(val_set, "Validation")
    test_combos = count_combinations(test_set, "Test")
    
    # Check if any combination is completely missing from train set
    all_combos = set(train_combos.keys()) | set(val_combos.keys()) | set(test_combos.keys())
    missing_in_train = all_combos - set(train_combos.keys())
    
    if missing_in_train:
        print(f"\nCRITICAL: {len(missing_in_train)} combinations missing from training set!")
        print("This will prevent the model from learning these personality types.")
    else:
        print("\nAll combinations present in training set")

def save_splits(
    train_set: List[Dict],
    val_set: List[Dict],
    test_set: List[Dict],
    output_dir: str = "data"
):
    """Save the split datasets to JSON files."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    splits = {
        'train_dataset.json': train_set,
        'validation_dataset.json': val_set,
        'test_dataset.json': test_set
    }
    
    for filename, dataset in splits.items():
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        print(f"✓ Saved {filename} ({len(dataset)} personas)")

def main():
    """Main execution"""
    print("="*60)
    print("Stratified Dataset Splitting")
    print("="*60)
    
    # Configuration
    SOURCE_PATH = 'data/big5-persona.json'  # Adjust as needed
    OUTPUT_DIR = 'data'
    SEED = 42
    
    # Check if source file exists
    if not os.path.exists(SOURCE_PATH):
        print(f"\nError: Source file not found: {SOURCE_PATH}")
        print("\nPlease update SOURCE_PATH in this script to point to your persona dataset.")
        return
    
    # Load personas
    print(f"\nLoading personas from: {SOURCE_PATH}")
    with open(SOURCE_PATH, 'r', encoding='utf-8') as f:
        all_personas = json.load(f)
    
    print(f"Loaded {len(all_personas)} personas")
    
    # Perform stratified split
    print("\nPerforming stratified split...")
    train_set, val_set, test_set = stratified_split_personas(
        all_personas,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        seed=SEED
    )
    
    # Verify stratification
    verify_stratification(train_set, val_set, test_set)
    
    # Save splits
    print("\n" + "="*60)
    print("Saving Split Datasets")
    print("="*60)
    save_splits(train_set, val_set, test_set, OUTPUT_DIR)
    
    print("\n" + "="*60)
    print("✓ Stratified splitting complete!")
    print("="*60)
    print("\nYou can now use these files with train.py:")
    print("  python train.py \\")
    print("    --train-dataset-path data/train_dataset.json \\")
    print("    --val-dataset-path data/validation_dataset.json \\")
    print("    --train-all")

if __name__ == '__main__':
    main()

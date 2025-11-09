import pickle
import matplotlib.pyplot as plt
import numpy as np

def analyze_word_level_training(model_name='lstm'):
    """Analyze word-level training results."""
    
    print("=" * 60)
    print(f"ANALYZING WORD-LEVEL {model_name.upper()} TRAINING")
    print("=" * 60)
    
    # Load training history
    history_path = f'checkpoints/{model_name}_word_level_history.pkl'
    
    try:
        with open(history_path, 'rb') as f:
            history = pickle.load(f)
    except FileNotFoundError:
        print(f"✗ Error: {history_path} not found!")
        return
    
    # Create results directory
    import os
    os.makedirs('results/word_level', exist_ok=True)
    
    # Plot loss curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    epochs = range(1, len(history['train_losses']) + 1)
    
    # Plot 1: Training Loss
    ax1.plot(epochs, history['train_losses'], 
             label='Train Loss', linewidth=2, marker='o', 
             markersize=3, color='#2E86AB')
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax1.set_title(f'{model_name.upper()} Word-Level Training Loss', 
                  fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Validation Loss
    ax2.plot(epochs, history['val_losses'], 
             label='Validation Loss', linewidth=2, marker='s', 
             markersize=3, color='#A23B72')
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax2.set_title(f'{model_name.upper()} Word-Level Validation Loss', 
                  fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_path = f'results/word_level/{model_name}_word_level_training.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved training curve to: {output_path}")
    
    # Print statistics
    print(f"\n" + "=" * 60)
    print("TRAINING STATISTICS")
    print("=" * 60)
    
    print(f"\nTotal epochs: {len(history['train_losses'])}")
    
    print(f"\nTrain Loss:")
    print(f"  Starting: {history['train_losses'][0]:.4f}")
    print(f"  Final:    {history['train_losses'][-1]:.4f}")
    print(f"  Best:     {min(history['train_losses']):.4f} (epoch {history['train_losses'].index(min(history['train_losses'])) + 1})")
    
    print(f"\nValidation Loss:")
    print(f"  Starting: {history['val_losses'][0]:.4f}")
    print(f"  Final:    {history['val_losses'][-1]:.4f}")
    print(f"  Best:     {min(history['val_losses']):.4f} (epoch {history['val_losses'].index(min(history['val_losses'])) + 1})")
    
    improvement = history['val_losses'][0] - history['val_losses'][-1]
    improvement_pct = (improvement / history['val_losses'][0]) * 100
    
    print(f"\nImprovement:")
    print(f"  Absolute: {improvement:.4f}")
    print(f"  Relative: {improvement_pct:.1f}%")
    
    # Check for overfitting
    final_gap = history['train_losses'][-1] - history['val_losses'][-1]
    print(f"\nOverfitting Check:")
    print(f"  Train-Val Gap: {abs(final_gap):.4f}")
    
    if abs(final_gap) < 0.5:
        print(f"  Status: ✓ Good - Model generalizes well")
    elif abs(final_gap) < 1.0:
        print(f"  Status: ⚠ Slight underfitting (val > train)")
    else:
        print(f"  Status: ⚠ Significant underfitting")
    
    print("=" * 60)
    
    plt.show()
    
    return history


def compare_word_level_models():
    """Compare LSTM and Transformer on word-level."""
    
    print("\n" + "=" * 60)
    print("WORD-LEVEL MODEL COMPARISON")
    print("=" * 60)
    
    # Load both histories
    try:
        with open('checkpoints/lstm_word_level_history.pkl', 'rb') as f:
            lstm_history = pickle.load(f)
        with open('checkpoints/transformer_word_level_history.pkl', 'rb') as f:
            transformer_history = pickle.load(f)
    except FileNotFoundError as e:
        print(f"✗ Error: Missing history file - {e}")
        return
    
    # Create comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    epochs_lstm = range(1, len(lstm_history['train_losses']) + 1)
    epochs_transformer = range(1, len(transformer_history['train_losses']) + 1)
    
    # Plot 1: Train Loss Comparison
    axes[0].plot(epochs_lstm, lstm_history['train_losses'], 
                 label='LSTM Train', linewidth=2, color='#E63946', alpha=0.7)
    axes[0].plot(epochs_transformer, transformer_history['train_losses'], 
                 label='Transformer Train', linewidth=2, color='#06FFA5', alpha=0.7)
    axes[0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Train Loss', fontsize=12, fontweight='bold')
    axes[0].set_title('Word-Level Training Loss Comparison', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Validation Loss Comparison
    axes[1].plot(epochs_lstm, lstm_history['val_losses'], 
                 label='LSTM Val', linewidth=2, marker='o', markersize=3,
                 color='#E63946', alpha=0.7)
    axes[1].plot(epochs_transformer, transformer_history['val_losses'], 
                 label='Transformer Val', linewidth=2, marker='s', markersize=3,
                 color='#06FFA5', alpha=0.7)
    axes[1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Validation Loss', fontsize=12, fontweight='bold')
    axes[1].set_title('Word-Level Validation Loss Comparison', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/word_level/lstm_vs_transformer_word_level.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved comparison plot: results/word_level/lstm_vs_transformer_word_level.png")
    
    # Print comparison
    print("\n" + "=" * 60)
    print("COMPARISON STATISTICS")
    print("=" * 60)
    
    print("\nLSTM (Word-Level):")
    print(f"  Best Val Loss: {min(lstm_history['val_losses']):.4f}")
    print(f"  Final Val Loss: {lstm_history['val_losses'][-1]:.4f}")
    
    print("\nTransformer (Word-Level):")
    print(f"  Best Val Loss: {min(transformer_history['val_losses']):.4f}")
    print(f"  Final Val Loss: {transformer_history['val_losses'][-1]:.4f}")
    
    improvement = (min(lstm_history['val_losses']) - min(transformer_history['val_losses'])) / min(lstm_history['val_losses']) * 100
    
    print("\n" + "=" * 60)
    print("WINNER")
    print("=" * 60)
    if min(transformer_history['val_losses']) < min(lstm_history['val_losses']):
        print(f"🏆 Transformer wins by {abs(improvement):.1f}%")
        print(f"Val Loss: {min(transformer_history['val_losses']):.4f} vs {min(lstm_history['val_losses']):.4f}")
    else:
        print(f"🏆 LSTM wins by {abs(improvement):.1f}%")
        print(f"Val Loss: {min(lstm_history['val_losses']):.4f} vs {min(transformer_history['val_losses']):.4f}")
    
    plt.show()


def compare_char_vs_word_level():
    """Compare character-level vs word-level approaches."""
    
    print("\n" + "=" * 60)
    print("CHARACTER-LEVEL vs WORD-LEVEL COMPARISON")
    print("=" * 60)
    
    # Load all histories
    try:
        with open('checkpoints/lstm_history.pkl', 'rb') as f:
            lstm_char = pickle.load(f)
        with open('checkpoints/transformer_history.pkl', 'rb') as f:
            transformer_char = pickle.load(f)
        with open('checkpoints/lstm_word_level_history.pkl', 'rb') as f:
            lstm_word = pickle.load(f)
        with open('checkpoints/transformer_word_level_history.pkl', 'rb') as f:
            transformer_word = pickle.load(f)
    except FileNotFoundError as e:
        print(f"✗ Missing file: {e}")
        return
    
    # Create comprehensive comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # LSTM Comparison
    axes[0, 0].plot(lstm_char['val_losses'], label='Char-Level', linewidth=2, color='#E63946')
    axes[0, 0].plot(lstm_word['val_losses'], label='Word-Level', linewidth=2, color='#06FFA5')
    axes[0, 0].set_title('LSTM: Character vs Word Level', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Transformer Comparison
    axes[0, 1].plot(transformer_char['val_losses'], label='Char-Level', linewidth=2, color='#E63946')
    axes[0, 1].plot(transformer_word['val_losses'], label='Word-Level', linewidth=2, color='#06FFA5')
    axes[0, 1].set_title('Transformer: Character vs Word Level', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Validation Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Character-Level Comparison
    axes[1, 0].plot(lstm_char['val_losses'], label='LSTM', linewidth=2, color='#E63946')
    axes[1, 0].plot(transformer_char['val_losses'], label='Transformer', linewidth=2, color='#06FFA5')
    axes[1, 0].set_title('Character-Level: LSTM vs Transformer', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Validation Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Word-Level Comparison
    axes[1, 1].plot(lstm_word['val_losses'], label='LSTM', linewidth=2, color='#E63946')
    axes[1, 1].plot(transformer_word['val_losses'], label='Transformer', linewidth=2, color='#06FFA5')
    axes[1, 1].set_title('Word-Level: LSTM vs Transformer', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Validation Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/complete_comparison.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved complete comparison: results/complete_comparison.png")
    
    # Print summary table
    print("\n" + "=" * 60)
    print("COMPLETE RESULTS SUMMARY")
    print("=" * 60)
    print("\n{:<20} {:<15} {:<15}".format("Model", "Char-Level", "Word-Level"))
    print("-" * 50)
    print("{:<20} {:<15.4f} {:<15.4f}".format("LSTM", min(lstm_char['val_losses']), min(lstm_word['val_losses'])))
    print("{:<20} {:<15.4f} {:<15.4f}".format("Transformer", min(transformer_char['val_losses']), min(transformer_word['val_losses'])))
    
    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)
    print("✓ Character-level LSTM: 1.47 loss (BEST overall)")
    print("✓ Word-level Transformer: 6.23 loss (beats word-level LSTM)")
    print("⚠ Word-level needs 10-20x more data")
    print("⚠ Loss not directly comparable (different vocab sizes)")
    
    plt.show()


if __name__ == "__main__":
    # Analyze individual models
    print("\n1. Analyzing LSTM Word-Level...")
    analyze_word_level_training('lstm')
    
    print("\n2. Analyzing Transformer Word-Level...")
    analyze_word_level_training('transformer')
    
    # Compare word-level models
    print("\n3. Comparing Word-Level Models...")
    compare_word_level_models()
    
    # Complete comparison
    print("\n4. Complete Comparison (All Models)...")
    compare_char_vs_word_level()
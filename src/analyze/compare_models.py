import pickle
import matplotlib.pyplot as plt
import numpy as np

def compare_models():
    """Compare RNN and LSTM training curves."""
    
    print("=" * 60)
    print("RNN vs LSTM COMPARISON")
    print("=" * 60)
    
    # Load both histories
    with open('checkpoints/simple_rnn_history.pkl', 'rb') as f:
        rnn_history = pickle.load(f)
    
    with open('checkpoints/lstm_history.pkl', 'rb') as f:
        lstm_history = pickle.load(f)
    
    # Create comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    epochs_rnn = range(1, len(rnn_history['train_losses']) + 1)
    epochs_lstm = range(1, len(lstm_history['train_losses']) + 1)
    
    # Plot 1: Train Loss Comparison
    axes[0].plot(epochs_rnn, rnn_history['train_losses'], 
                 label='RNN Train', linewidth=2, color='#E63946', alpha=0.7)
    axes[0].plot(epochs_lstm, lstm_history['train_losses'], 
                 label='LSTM Train', linewidth=2, color='#06FFA5', alpha=0.7)
    axes[0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Train Loss', fontsize=12, fontweight='bold')
    axes[0].set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Validation Loss Comparison
    axes[1].plot(epochs_rnn, rnn_history['val_losses'], 
                 label='RNN Val', linewidth=2, marker='o', markersize=3,
                 color='#E63946', alpha=0.7)
    axes[1].plot(epochs_lstm, lstm_history['val_losses'], 
                 label='LSTM Val', linewidth=2, marker='s', markersize=3,
                 color='#06FFA5', alpha=0.7)
    axes[1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Validation Loss', fontsize=12, fontweight='bold')
    axes[1].set_title('Validation Loss Comparison', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/rnn_vs_lstm_comparison.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved comparison plot: results/rnn_vs_lstm_comparison.png")
    
    # Print statistics
    print("\n" + "=" * 60)
    print("STATISTICS")
    print("=" * 60)
    
    print("\nSimple RNN:")
    print(f"  Best Val Loss: {min(rnn_history['val_losses']):.4f}")
    print(f"  Final Val Loss: {rnn_history['val_losses'][-1]:.4f}")
    print(f"  Parameters: 273,905")
    
    print("\nLSTM:")
    print(f"  Best Val Loss: {min(lstm_history['val_losses']):.4f}")
    print(f"  Final Val Loss: {lstm_history['val_losses'][-1]:.4f}")
    print(f"  Parameters: 3,765,105")
    
    improvement = (min(rnn_history['val_losses']) - min(lstm_history['val_losses'])) / min(rnn_history['val_losses']) * 100
    
    print("\n" + "=" * 60)
    print("IMPROVEMENT")
    print("=" * 60)
    print(f"LSTM is {improvement:.1f}% better than RNN")
    print(f"Val Loss: {min(rnn_history['val_losses']):.4f} → {min(lstm_history['val_losses']):.4f}")
    
    plt.show()

if __name__ == "__main__":
    compare_models()
import pickle
import matplotlib.pyplot as plt
import os

def analyze_training(model_name='simple_rnn', show_plot=True):
    """
    Analyze and visualize training results.
    
    Args:
        model_name: Name of the model (simple_rnn, lstm, transformer)
        show_plot: Whether to display the plot interactively
    """
    
    print("=" * 60)
    print(f"ANALYZING {model_name.upper()} TRAINING")
    print("=" * 60)
    
    # Load training history
    history_path = f'checkpoints/{model_name}_history.pkl'
    
    if not os.path.exists(history_path):
        print(f"✗ Error: {history_path} not found!")
        print("  Make sure you've trained the model first.")
        return
    
    with open(history_path, 'rb') as f:
        history = pickle.load(f)
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Plot loss curves
    plt.figure(figsize=(12, 7))
    
    epochs = range(1, len(history['train_losses']) + 1)
    
    plt.plot(epochs, history['train_losses'], 
             label='Train Loss', linewidth=2, marker='o', 
             markersize=3, color='#2E86AB')
    plt.plot(epochs, history['val_losses'], 
             label='Validation Loss', linewidth=2, marker='s', 
             markersize=3, color='#A23B72')
    
    plt.xlabel('Epoch', fontsize=14, fontweight='bold')
    plt.ylabel('Loss (Cross Entropy)', fontsize=14, fontweight='bold')
    plt.title(f'{model_name.upper()} Training Progress', 
              fontsize=16, fontweight='bold', pad=20)
    plt.legend(fontsize=12, loc='best')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    # Save plot
    output_path = f'results/{model_name}_training_curve.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved training curve to: {output_path}")
    
    # Print detailed statistics
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
    
    if abs(final_gap) < 0.1:
        print(f"  Status: ✓ Good - Model is well-balanced")
    elif abs(final_gap) < 0.3:
        print(f"  Status: ⚠ Slight overfitting")
    else:
        print(f"  Status: ✗ Overfitting detected")
    
    print("=" * 60)
    
    # Show plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return history


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
    else:
        model_name = 'simple_rnn'
    
    analyze_training(model_name, show_plot=True)
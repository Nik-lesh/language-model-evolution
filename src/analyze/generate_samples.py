import torch
import pickle
import sys
sys.path.append('src')
from models.simple_rnn import SimpleRNN
from models.lstm import LSTMModel

def load_model(model_path, model_type='rnn'):
    """Load a trained model."""
    checkpoint = torch.load(model_path, map_location='cpu')
    
    if model_type == 'rnn':
        model = SimpleRNN(
            vocab_size=checkpoint['vocab_size'],
            embedding_dim=checkpoint.get('embedding_dim', 128),
            hidden_dim=checkpoint.get('hidden_dim', 256),
            num_layers=checkpoint.get('num_layers', 2)
        )
    else:
        model = LSTMModel(
            vocab_size=checkpoint['vocab_size'],
            embedding_dim=checkpoint.get('embedding_dim', 256),
            hidden_dim=checkpoint.get('hidden_dim', 512),
            num_layers=checkpoint.get('num_layers', 2)
        )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model

def generate_text_manual(model, dataset_dict, start_text, length=300, temperature=0.8):
    """
    Generate text manually without using model.generate()
    This avoids the dict vs object issue.
    """
    model.eval()
    device = next(model.parameters()).device
    
    # Get mappings from dict
    char_to_idx = dataset_dict['char_to_idx']
    idx_to_char = dataset_dict['idx_to_char']
    
    # Encode starting text
    chars = [char_to_idx[ch] for ch in start_text]
    input_seq = torch.LongTensor(chars).unsqueeze(0).to(device)
    
    generated = start_text
    hidden = None
    
    with torch.no_grad():
        for _ in range(length):
            # Forward pass
            output, hidden = model(input_seq, hidden)
            
            # Get last output and apply temperature
            output = output[-1] / temperature
            
            # Convert to probabilities
            probs = torch.softmax(output, dim=0)
            
            # Sample next character
            next_char_idx = torch.multinomial(probs, 1).item()
            next_char = idx_to_char[next_char_idx]
            
            generated += next_char
            
            # Next input
            input_seq = torch.LongTensor([[next_char_idx]]).to(device)
    
    return generated

def compare_generations():
    """Compare text generation from both models."""
    
    print("=" * 60)
    print("TEXT GENERATION COMPARISON")
    print("=" * 60)
    
    # Load dataset
    with open('data/dataset.pkl', 'rb') as f:
        dataset_dict = pickle.load(f)
    
    # Load models
    print("\nLoading models...")
    rnn = load_model('checkpoints/simple_rnn_best.pth', 'rnn')
    lstm = load_model('checkpoints/lstm_best.pth', 'lstm')
    print("✓ Models loaded")
    
    # Test prompts
    prompts = [
        "Money is",
        "Investing in",
        "The wealthy",
        "Financial freedom"
    ]
    
    for prompt in prompts:
        print("\n" + "=" * 60)
        print(f"PROMPT: '{prompt}'")
        print("=" * 60)
        
        print("\n[SIMPLE RNN]")
        print("-" * 60)
        rnn_text = generate_text_manual(rnn, dataset_dict, prompt, length=200, temperature=0.8)
        print(rnn_text)
        
        print("\n[LSTM]")
        print("-" * 60)
        lstm_text = generate_text_manual(lstm, dataset_dict, prompt, length=200, temperature=0.8)
        print(lstm_text)
    
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)
    print("\nKey Observations:")
    print("   Compare word formation quality")
    print("   Check sentence coherence")
    print("   Evaluate context retention")
    print("   Assess financial terminology usage")
    print("\nLSTM should show:")
    print("   Better word formation (fewer made-up words)")
    print("   More coherent sentence structure")
    print("   Better context retention")
    print("   More realistic financial terminology")

if __name__ == "__main__":
    compare_generations()
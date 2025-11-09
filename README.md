# Language Model Evolution: RNN → LSTM → Transformer

Comparing three generations of language models trained on the same financial text corpus.

## 🎯 Project Goal

Train and compare three different neural network architectures for character-level language modeling:

1. **Simple RNN** (baseline) - Shows fundamental sequence modeling
2. **LSTM** (improved) - Demonstrates handling of long-term dependencies
3. **Transformer** (state-of-the-art) - Modern attention-based architecture

All models trained on finance/investment books to learn domain-specific language.

## 📊 Dataset

- **Domain:** Finance and Investment
- **Size:** ~640 KB of text (~652,809 characters)
- **Vocabulary:** 113 unique characters
- **Books:** 2 finance books focused on money, wealth, and investing
- **Preprocessing:** Character-level tokenization
- **Train/Val Split:** 90/10

## 🏗️ Project Structure

```
language-model-evolution/
├── data/                          # Training data (not in git)
│   ├── training_corpus.txt        # Raw combined text
│   ├── training_corpus_clean.txt  # Cleaned text
│   └── dataset.pkl                # Processed dataset
│
├── pdfs/                          # Original PDF books (not in git)
│
├── src/
│   ├── scripts/                   # Data processing scripts
│   │   ├── extract_from_pdf.py    # PDF → text extraction
│   │   ├── clean_corpus.py        # Text cleaning
│   │   ├── analysis_corpus.py     # Dataset statistics
│   │   └── prepare_data.py        # Create training sequences
│   │
│   ├── models/                    # Model architectures
│   │   ├── simple_rnn.py          # Basic RNN implementation
│   │   ├── lstm.py                # LSTM implementation
│   │   └── transformer.py         # Transformer implementation
│   │
│   ├── train.py                   # Training script
│   ├── analyze_training.py        # Training visualization
│   ├── compare_models.py          # Model comparison
│   ├── generate_samples.py        # Text generation comparison
│   └── create_report.py           # Report generation
│
├── checkpoints/                   # Saved models (not in git)
├── results/                       # Training curves and analysis
├── requirements.txt               # Python dependencies
├── .gitignore
└── README.md
```

## 🚀 Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/language-model-evolution.git
cd language-model-evolution
```

### 2. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare your data

Since the PDFs are copyrighted, you'll need to provide your own:

- Place PDF files in `pdfs/` folder
- Run data extraction: `python src/scripts/extract_from_pdf.py`
- Clean and prepare: `python src/scripts/prepare_data.py`

## 📦 Dependencies

```
torch>=2.0.0
numpy>=1.24.0
pdfplumber>=0.11.0
PyPDF2>=3.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
pandas>=2.0.0
jupyter>=1.0.0
tqdm>=4.65.0
```

## 🎓 Models

### Simple RNN

- **Parameters:** 273,905
- **Architecture:** Embedding(113, 128) → RNN(128, 256, 2 layers) → Linear(256, 113)
- **Training:** 50 epochs, 3.82 minutes
- **Val Loss:** 1.6482

### LSTM

- **Parameters:** 3,765,105 (13.7x more than RNN)
- **Architecture:** Embedding(113, 256) → LSTM(256, 512, 2 layers) → Linear(512, 113)
- **Training:** 50 epochs, 39.45 minutes
- **Val Loss:** 1.4711 (10.7% better than RNN)

### Transformer

- **Parameters:** 3,217,009
- **Architecture:** Embedding(113, 256) → 4-layer Transformer (8 heads) → Linear(256, 113)
- **Training:** 50 epochs, 102.96 minutes (v1), 100 epochs (v2 in progress)
- **Val Loss:** 1.5523 (v1), TBD (v2)
- **Features:** Multi-head attention, positional encoding, causal masking

## 📈 Training

```bash
# Train Simple RNN
python src/train.py rnn

# Train LSTM
python src/train.py lstm

# Train Transformer
python src/train.py transformer
```

## 🏆 Results Comparison

### Quantitative Metrics

| Model          | Parameters | Val Loss   | Train Time | Improvement       |
| -------------- | ---------- | ---------- | ---------- | ----------------- |
| Simple RNN     | 273,905    | 1.6482     | 3.8 min    | Baseline          |
| **LSTM** 🥇    | 3,765,105  | **1.4711** | 39.5 min   | **10.7% better**  |
| Transformer v1 | 3,217,009  | 1.5523     | 103 min    | Needs more epochs |
| Transformer v2 | 3,217,009  | TBD        | ~100 min   | In progress       |

### Qualitative Comparison - Text Generation

**Prompt:** "Money is"

**Simple RNN:**

```
Money is important on the first for investors of counting. And for
successe of ommerial peaper of investors...
```

❌ Made-up words: "ommerial peaper", "successe"  
❌ Poor grammar and incoherent structure

**LSTM (Winner):**

```
Money is always right to seek by less than investments. When the
result is that the poor and the drivers that are high-specialized...
```

✅ All real words  
✅ Financial vocabulary: "investments", "poor", "specialized"  
✅ Better grammar and sentence structure

**Transformer v1:**

```
Money is f t thecathe Couifthisie atr, pere ak t'vente nado...
```

⚠️ Undertrained - needs more epochs (still improving at cutoff)

### Sample Generations from All Prompts

| Prompt              | RNN                      | LSTM                                        | Winner  |
| ------------------- | ------------------------ | ------------------------------------------- | ------- |
| "Money is"          | "ommerial peaper"        | "investments... poor... specialized"        | LSTM 🥇 |
| "Investing in"      | "gots surcession"        | "retirement traders... 20%... difference"   | LSTM 🥇 |
| "The wealthy"       | "smort... servically"    | "company... management... risk... expenses" | LSTM 🥇 |
| "Financial freedom" | "maxiess... comparation" | "$17,000 a month... investment experience"  | LSTM 🥇 |

## 🔍 Key Findings

### 1. LSTM Clearly Outperforms Simple RNN

- **10.7% better validation loss** (1.4711 vs 1.6482)
- **Generates real words** vs RNN's made-up words
- **Better context retention** over longer sequences
- **Domain knowledge:** Uses financial terms appropriately

### 2. LSTM's Success Factors

- **Memory gates** control information flow effectively
- **Cell state** maintains long-term context
- **Prevents vanishing gradients** that plague simple RNNs
- **13.7x more parameters** but worth the training time

### 3. Transformer Insights

- **Architecture works** (37.7% improvement from start)
- **Still improving at epoch 50** (needs more training)
- **More complex** = needs more epochs to converge
- **Small datasets** may favor LSTM over Transformer

### 4. Text Generation Quality

**Word Formation:**

- RNN: Frequent gibberish ("ommerial", "maxiess", "comparation")
- LSTM: Almost exclusively real words
- Winner: LSTM 🥇

**Grammar & Syntax:**

- RNN: Broken sentence structure
- LSTM: Maintains grammatical structure across clauses
- Winner: LSTM 🥇

**Domain Knowledge:**

- RNN: Some financial terms but inconsistent
- LSTM: Consistent financial vocabulary, even generates dollar amounts!
- Winner: LSTM 🥇

### 5. Impressive LSTM Achievements

✅ Generated specific numbers: "$17,000 a month"  
✅ Complex financial concepts: "retirement traders", "investment experience"  
✅ Subject-verb agreement over multiple words  
✅ Multi-clause sentences with logical connections

## 📊 Visualizations

Training curves and comparisons available in `results/`:

- `simple_rnn_training_curve.png`
- `lstm_training_curve.png`
- `transformer_training_curve.png`
- `rnn_vs_lstm_comparison.png`

## 📝 Technical Analysis

### Why LSTM Performs Better Than RNN

1. **Memory Cells:** Long-term memory via cell state
2. **Gating Mechanisms:**
   - Forget Gate: Decides what to discard
   - Input Gate: Controls new information storage
   - Output Gate: Determines output from cell state
3. **Gradient Flow:** Architecture prevents vanishing gradients
4. **Capacity:** More parameters enable complex pattern learning

### Trade-offs

**RNN Advantages:**

- ✅ 10x faster training
- ✅ 13.7x fewer parameters
- ✅ Lower memory requirements
- ✅ Good for quick prototyping

**LSTM Advantages:**

- ✅ 10.7% better loss
- ✅ Significantly better text quality
- ✅ Better long-range dependencies
- ✅ More coherent output
- ✅ Domain-appropriate vocabulary

**Transformer Considerations:**

- ✅ Parallel processing capability
- ✅ Self-attention mechanism
- ⚠️ Needs more training time
- ⚠️ May need more data for full potential

## 🚀 Usage Examples

### Generate Text

```python
from models.lstm import LSTMModel
import pickle

# Load model
model = LSTMModel.load('checkpoints/lstm_best.pth')

# Load dataset
with open('data/dataset.pkl', 'rb') as f:
    dataset = pickle.load(f)

# Generate
text = model.generate(dataset, "Money is", length=200, temperature=0.8)
print(text)
```

### Analyze Training

```bash
python src/analyze_training.py lstm
```

### Compare Models

```bash
python src/compare_models.py
python src/generate_samples.py
```

## 📚 Key Learnings

1. **Architecture evolution matters:** RNN → LSTM shows clear improvement
2. **Gates are crucial:** LSTM's gating mechanism dramatically improves performance
3. **Context is king:** Better long-term memory = better text generation
4. **Training time trade-offs:** 10x slower training worth it for quality
5. **Domain adaptation works:** Models learned financial vocabulary effectively
6. **Hyperparameter tuning:** Different architectures need different settings
7. **Small data insights:** LSTM may outperform Transformer on limited data

## 🎯 Future Improvements

- [ ] Complete Transformer v2 training (100 epochs)
- [ ] Implement beam search for better generation
- [ ] Try word-level tokenization
- [ ] Scale to larger dataset (5-10MB)
- [ ] Add temperature and top-k sampling experiments
- [ ] Implement Transformer decoder architecture
- [ ] Fine-tune on specific financial domains

## 🤝 Contributing

This is a learning project demonstrating language model evolution. Feel free to:

- Fork and experiment with different architectures
- Try different datasets
- Implement additional models (GRU, Transformer-XL, etc.)
- Add evaluation metrics (perplexity, BLEU)

## 📄 License

MIT License - See LICENSE file for details

## 🙏 Acknowledgments

- Dataset: Finance and investment books
- Inspired by: The evolution of NLP architectures from RNN to Transformers
- Built for: Understanding deep learning fundamentals and architectural improvements
- Frameworks: PyTorch, NumPy, Matplotlib

## 📞 Contact

**Project:** Language Model Evolution  
**Purpose:** Educational demonstration of RNN → LSTM → Transformer progression  
**Status:** ✅ RNN Complete | ✅ LSTM Complete | 🔄 Transformer v2 In Progress

---

**Last Updated:** November 8, 2025  
**Current Best Model:** LSTM (1.4711 val loss)  
**Training Status:** Transformer v2 retraining with 100 epochs for improved performance

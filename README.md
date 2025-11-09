# Language Model Evolution: RNN → LSTM → Transformer

Comparing three generations of language models trained on financial text to demonstrate architectural improvements in deep learning.

## 🎯 Project Overview

Built and trained three neural network architectures from scratch:

- **Simple RNN** - Baseline sequential model
- **LSTM** - Improved with memory gates
- **Transformer** - State-of-the-art attention mechanism

**Dataset:** 640KB financial text (character-level tokenization)  
**Goal:** Compare text generation quality and training efficiency

## 🏆 Results

| Model       | Parameters | Val Loss   | Train Time | Text Quality      |
| ----------- | ---------- | ---------- | ---------- | ----------------- |
| Simple RNN  | 274K       | 1.6482     | 3.8 min    | Poor (gibberish)  |
| **LSTM** 🥇 | 3.8M       | **1.4711** | 39.5 min   | **Excellent**     |
| Transformer | 3.2M       | 1.5523     | 103 min    | Needs improvement |

### Sample Generation: "Money is..."

**RNN:** "Money is important on the first for investors of counting. And for successe of ommerial peaper..."  
❌ Made-up words, broken grammar

**LSTM:** "Money is always right to seek by less than investments. When the result is that the poor and the drivers that are high-specialized..."  
✅ Real words, financial vocabulary, coherent structure

**Transformer:** "Money is f t thecathe Couifthisie atr, pere ak..."  
⚠️ Undertrained on small dataset

### Key Finding

**LSTM wins** for character-level modeling on small datasets. Demonstrates that newer architectures aren't always better - match your model to your data!

## 🚀 Quick Start

```bash
# Setup
git clone https://github.com/YOUR_USERNAME/language-model-evolution.git
cd language-model-evolution
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Train models
python src/train.py rnn        # 4 min
python src/train.py lstm       # 40 min
python src/train.py transformer # 100 min

# Compare results
python src/compare_models.py
python src/generate_samples.py
```

## 📁 Project Structure

```
language-model-evolution/
├── src/
│   ├── models/          # RNN, LSTM, Transformer implementations
│   ├── scripts/         # Data processing (PDF → text)
│   └── train.py         # Training pipeline
├── results/             # Training curves and visualizations
├── data/                # Training corpus (not in git)
└── checkpoints/         # Saved models (not in git)
```

## 🔬 Technical Details

**Architecture Highlights:**

- **RNN:** Simple recurrent connections, struggles with long-term dependencies
- **LSTM:** Gates (forget/input/output) + cell state for better memory
- **Transformer:** Multi-head self-attention (8 heads, 4 layers) with positional encoding

**Training Setup:**

- Optimizer: Adam
- Batch size: 64 (RNN/LSTM), 32 (Transformer)
- Sequence length: 100 characters
- Loss: Cross-entropy

## 📊 Visualizations

See `results/` for:

- Individual training curves
- Side-by-side model comparison
- Text generation examples

## 🚀 Future Improvements

**Phase 1:** Word-level tokenization (in progress)

- Switch from 113 characters → 10K word vocabulary
- Expected to dramatically improve Transformer performance

**Phase 2:** Expand dataset to 50-100 books

- Current: 2 books (640KB) → Target: 10-50MB
- More data = better models, especially Transformer

**Phase 3:** GPU training on Google Colab

- 10-50x faster training
- Enables larger models and rapid experimentation

**Phase 4:** Production financial advisor chatbot

- GPT-style interface for investment advice
- FastAPI backend + React frontend
- Deployment ready

## 📚 Key Learnings

1. **Bigger ≠ always better** - LSTM beat Transformer on this dataset
2. **Architecture matters** - Gates solve vanishing gradients
3. **Data size is crucial** - Transformers need more data to shine
4. **Domain adaptation works** - Models learned financial vocabulary
5. **Training trade-offs** - 10x time for 10.7% improvement worth it

## 🛠️ Tech Stack

- PyTorch 2.0
- NumPy, Matplotlib
- pdfplumber (data extraction)
- tqdm (progress bars)

## 📄 License

MIT License

---

**Status:** ✅ RNN Complete | ✅ LSTM Complete | 🔄 Transformer Optimization  
**Best Model:** LSTM (1.4711 val loss)  
**Next:** Word-level tokenization + expanded dataset

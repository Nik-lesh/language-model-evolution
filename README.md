# Language Model Evolution: RNN → LSTM → Transformer

Comparing three generations of language models trained on financial text, demonstrating architectural improvements and the critical impact of tokenization strategy and dataset size.

## 🎯 Project Overview

Built and trained three neural network architectures from scratch on two tokenization approaches:

- **Simple RNN** - Baseline sequential model
- **LSTM** - Improved with memory gates
- **Transformer** - State-of-the-art attention mechanism

**Tokenization Strategies:**

- Character-level (113 vocab)
- Word-level (9,751 vocab)

**Goal:** Compare architectures and understand when each excels

## 🏆 Results

### Character-Level Tokenization (640KB, 652K characters)

| Model       | Parameters | Val Loss   | Train Time | Text Quality     |
| ----------- | ---------- | ---------- | ---------- | ---------------- |
| Simple RNN  | 274K       | 1.6482     | 3.8 min    | Poor (gibberish) |
| **LSTM** 🥇 | 3.8M       | **1.4711** | 39.5 min   | **Excellent**    |
| Transformer | 3.2M       | 1.5523     | 103 min    | Undertrained     |

### Word-Level Tokenization (640KB, 132K words)

| Model              | Parameters | Val Loss   | Train Time | Text Quality                 |
| ------------------ | ---------- | ---------- | ---------- | ---------------------------- |
| LSTM               | 3.8M       | 6.4072     | 15.7 min   | Good grammar, weak semantics |
| **Transformer** 🥇 | 3.2M       | **6.2291** | 25.8 min   | Similar, slightly better     |

### Sample Generations: "Money is..."

**Character-Level LSTM (Best Overall):**

```
"Money is always right to seek by less than investments. When the
result is that the poor and the drivers that are high-specialized..."
```

✅ Real words, financial vocabulary, semantic coherence

**Word-Level Transformer:**

```
"money is all, the road and white kahneman hurried off the second
investor, but if you want to be so losing money at some three..."
```

✅ Perfect grammar, learned names (Kahneman)  
❌ Semantically incoherent (insufficient data)

## 🔍 Key Findings

### 1. Character-Level: LSTM Wins

- **Best validation loss:** 1.4711
- **Best text quality** for small datasets
- **Optimal for:** <1MB text, character-level modeling
- Generated financial concepts: "$17,000 a month", "retirement traders"

### 2. Word-Level Requires More Data

- **Both models struggled** (6.2-6.4 loss vs 1.5 loss)
- **Vocabulary too sparse:** 9,751 words / 132K total = 13.6 samples per word
- **Transformer beats LSTM** at word-level (6.23 vs 6.41)
- **Conclusion:** Need 10-20x more data for word-level to work

### 3. Critical Insight: Data Size Matters More Than Architecture

- Small dataset (640KB) → Character-level dominates
- Large dataset (5-10MB) → Word-level should dominate
- **Architecture choice depends on data availability**

### 4. Tokenization Strategy Impact

| Aspect                        | Character-Level | Word-Level           |
| ----------------------------- | --------------- | -------------------- |
| **Vocabulary**                | 113             | 9,751                |
| **Samples per token**         | 5,775           | 13.6                 |
| **Best for**                  | Small datasets  | Large datasets       |
| **Training speed**            | Slower          | Faster               |
| **Text quality (small data)** | Better          | Worse                |
| **Text quality (large data)** | Good            | Excellent (expected) |

## 🚀 Quick Start

```bash
# Setup
git clone https://github.com/YOUR_USERNAME/language-model-evolution.git
cd language-model-evolution
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Character-level training
python src/train.py lstm           # 40 min
python src/train.py transformer    # 100 min

# Word-level training
python src/scripts/prepare_word_level_data.py
python src/train_word_level.py lstm        # 16 min
python src/train_word_level.py transformer # 26 min

# Compare results
python src/compare_models.py
python src/generate_samples.py
```

## 📁 Project Structure

```
language-model-evolution/
├── data/                          # Training data (not in git)
│   ├── books/                    # Source books by category
│   │   ├── gutenberg/            # 60 original classics (34 MB)
│   │   ├── gutenberg_expanded/   # 169 additional books (73 MB)
│   │   ├── wikipedia/            # 127 articles (4.6 MB)
│   │   ├── academic_text/        # 10 extracted papers (1.3 MB)
│   │   └── old_books/            # Original 2 books (0.6 MB)
│   ├── mega_corpus.txt           # Combined corpus (103 MB, 18.3M words)
│   ├── dataset.pkl               # Character-level dataset (652K chars)
│   └── word_dataset.pkl          # Word-level dataset (small corpus)
│
├── pdfs/                          # Original PDFs (not in git)
│
├── src/
│   ├── models/                   # Neural network architectures
│   │   ├── simple_rnn.py         # Simple RNN implementation
│   │   ├── lstm.py               # LSTM with gates
│   │   └── transformer.py        # Multi-head attention
│   │
│   ├── scripts/                  # Data processing utilities
│   │   ├── download_data/        # Data collection scripts (local only)
│   │   └── utils/                # Data preparation utilities
│   │       ├── prepare_data.py              # Character-level prep
│   │       └── prepare_word_level_data.py   # Word-level prep
│   │
│   ├── analyze/                  # Analysis and visualization
│   │   ├── analysis_corpus.py
│   │   ├── analyze_training.py
│   │   ├── analyze_word_level.py
│   │   ├── compare_models.py
│   │   └── generate_samples.py
│   │
│   ├── train/                    # Training scripts
│   │   ├── train.py              # Character-level training
│   │   └── train_word_level.py   # Word-level training
│   │
│   └── clean_corpus.py           # Text cleaning utility
│
├── results/                      # Training visualizations (in git)
│   ├── word_level/               # Word-level experiment results
│   ├── simple_rnn_training_curve.png
│   ├── lstm_training_curve.png
│   ├── transformer_training_curve.png
│   └── rnn_vs_lstm_comparison.png
│
├── checkpoints/                  # Trained models (not in git)
│
├── requirements.txt              # Python dependencies
├── .gitignore
└── README.md
```

## 📊 Dataset Summary

### Current Dataset (103 MB - Phase 2 Complete)

- **Total Size:** 103.14 MB
- **Total Words:** 18,330,423 (138x increase from original)
- **Total Sources:** 379 books and articles
- **Vocabulary:** ~20,000 unique words
- **Samples per Word:** ~916 (vs 13.6 in small dataset)

### Sources Breakdown

| Source             | Files   | Size       | Content                          |
| ------------------ | ------- | ---------- | -------------------------------- |
| Gutenberg Original | 60      | 31 MB      | Classic finance books            |
| Gutenberg Expanded | 169     | 66 MB      | Economics, investment, business  |
| Wikipedia          | 127     | 4.3 MB     | Finance/economics articles       |
| Academic Papers    | 10      | 1.3 MB     | Research papers (arXiv)          |
| **Total**          | **379** | **103 MB** | **Comprehensive finance corpus** |

### Categories Covered

- 📚 Economics theory and history
- 💹 Investment and trading strategies
- 🏢 Business and entrepreneurship
- 💰 Personal finance and wealth building
- 🏛️ Banking and monetary systems
- 📊 Accounting and financial management
- 🌍 International trade and economics
- 👔 Labor and industrial relations

## 🚀 Usage

### Prepare Datasets

**Character-level (small corpus):**

```bash
python src/scripts/utils/prepare_data.py
```

**Word-level (mega corpus):**

```bash
python src/scripts/utils/prepare_word_level_data.py data/mega_corpus.txt
```

### Train Models

**Character-level:**

```bash
python src/train/train.py rnn
python src/train/train.py lstm
python src/train/train.py transformer
```

**Word-level (on mega corpus):**

```bash
python src/train/train_word_level.py lstm
python src/train/train_word_level.py transformer
```

### Analyze Results

```bash
python src/analyze/analyze_training.py lstm
python src/analyze/compare_models.py
python src/analyze/generate_samples.py
```

## 📈 Expected Results with Mega Corpus

| Model                | Small Dataset | Mega Dataset      | Improvement        |
| -------------------- | ------------- | ----------------- | ------------------ |
| **Char LSTM**        | 1.47 loss     | N/A               | Baseline           |
| **Word LSTM**        | 6.41 loss     | ~1.8-2.2 loss     | 3-4x better        |
| **Word Transformer** | 6.23 loss     | **~1.2-1.8 loss** | **3-5x better** 🎯 |

**With 18.3M words, Transformer should dominate!**

## 🔬 Technical Details

**Character-Level Architectures:**

- **RNN:** Embedding(113, 128) → RNN(128, 256, 2 layers) → Linear(256, 113)
- **LSTM:** Embedding(113, 256) → LSTM(256, 512, 2 layers) → Linear(512, 113)
- **Transformer:** Embedding(113, 256) → 4-layer (8 heads) → Linear(256, 113)

**Word-Level Architectures:**

- **Vocabulary:** 9,751 words (top 10K, 100% coverage)
- **Sequence length:** 50 words (vs 100 chars)
- **Same model architectures**, different vocab size

**Training Details:**

- Optimizer: Adam
- Batch size: 64 (char), 32 (word)
- Loss: Cross-entropy
- Hardware: CPU (Apple Silicon)

## 📊 Visualizations

See `results/` for:

- Training curves (all models)
- Character vs word-level comparison
- Text generation examples

## 🚀 Next Steps: Phase 2 (In Progress)

### Current Status

✅ Proved word-level needs more data  
✅ Transformer beats LSTM at word-level  
🔄 Collecting 50-100 finance books

### Phase 2: Massive Dataset Expansion

**Goal:** 5-10MB corpus (10-20x larger)

**Books Sources:**

- Project Gutenberg economics classics (13+ books)
- Federal Reserve publications
- IMF/World Bank reports
- Modern finance bestsellers (library/purchase)

**Expected Impact:**

- Word vocabulary: 15-20K words
- Total words: 2-5M (vs current 132K)
- Word-level loss: < 2.0 (vs current 6.2)
- **Transformer should dominate**

### Phase 3: GPU Training

- Platform: Google Colab (free T4 GPU)
- Speed: 10-50x faster training
- Enables: Larger models, rapid experimentation

### Phase 4: Production Financial Advisor

- GPT-style chat interface
- FastAPI + React deployment
- Real-time financial advice generation

## 📚 Key Learnings

1. **Data size trumps architecture** - Match your approach to your data
2. **Character-level works great for small datasets** - Don't underestimate simplicity
3. **Word-level requires 10-20x more data** - Vocabulary sparsity is critical
4. **Transformers need proper conditions** - Not universally superior
5. **LSTM remains powerful** - Still competitive for many tasks
6. **Tokenization matters as much as architecture** - Choose wisely

## 🛠️ Tech Stack

- PyTorch 2.0
- NumPy, Matplotlib
- pdfplumber (PDF extraction)
- tqdm (progress tracking)

## 📊 Comparison Summary

| Scenario                            | Winner                 | Reason                     |
| ----------------------------------- | ---------------------- | -------------------------- |
| **Small dataset + character-level** | LSTM                   | Optimal samples per token  |
| **Small dataset + word-level**      | Transformer            | But both perform poorly    |
| **Large dataset + word-level**      | Transformer (expected) | Attention shines with data |

## 📄 License

MIT License

## 🙏 Acknowledgments

- Dataset: Finance books (Rich Dad Poor Dad, Psychology of Money)
- Inspired by: Evolution of NLP architectures
- Built for: Understanding when each architecture excels

---

**Current Status:**  
✅ **Phase 1 Complete** - Word-level validation  
🔄 **Phase 2 In Progress** - Dataset expansion (targeting 50-100 books)  
⏳ **Phase 3 Planned** - GPU training  
⏳ **Phase 4 Planned** - Production deployment

**Key Takeaway:** Small dataset (640KB) → Character-level LSTM wins (1.47 loss)  
**Next Goal:** Large dataset (5-10MB) → Word-level Transformer should dominate

**Last Updated:** November 9, 2025

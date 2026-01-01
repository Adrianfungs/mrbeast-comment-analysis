# MrBeast Comment Sentiment Analysis (TPU Accelerated)

## 📌 Project Overview
This project performs **Sentiment Analysis** on YouTube comments from MrBeast's channel, classifying them as **Positive**, **Neutral**, or **Negative**. 

Unlike standard sentiment analysis projects, this implementation is optimized for **High-Performance Computing (HPC)**. It utilizes **Google Cloud TPUs (Tensor Processing Units)** via `torch_xla` to accelerate the fine-tuning of a BERT-based model. It demonstrates proficiency in distributed training, handling class imbalance, and deploying Transformer models.

## 🚀 Key Technical Features
* **TPU Acceleration:** Leverages `torch_xla` and `xmp.spawn` to distribute training across TPU cores, significantly reducing training time compared to standard CPU/GPU workflows.
* **Transformer Architecture:** Fine-tunes the `tabularisai/multilingual-sentiment-analysis` BERT model for specific domain adaptation.
* **Imbalanced Data Handling:** Implements **Weighted Cross-Entropy Loss** (`class_weights_tensor`) to prevent bias toward the majority class (Positive comments).
* **Distributed Data Loading:** Uses `DistributedSampler` and `ParallelLoader` to efficiently feed data to multiple TPU cores simultaneously.
* **Robust Evaluation:** Calculates Accuracy, F1-Score (Macro), Precision, and Recall to give a complete picture of model performance.

## 🛠️ Tech Stack
* **Language:** Python 3.10+
* **Deep Learning:** PyTorch, PyTorch XLA (for TPU support)
* **NLP:** Hugging Face Transformers (`AutoTokenizer`, `AutoModelForSequenceClassification`)
* **Data Processing:** Pandas, NumPy, Scikit-learn
* **Environment:** Kaggle / Google Colab (TPU v3-8)

## 📊 Dataset
* **Source:** MrBeast YouTube Comment Analysis Dataset (Kaggle).
* **Input:** Raw comment text.
* **Labels:** mapped to `0: Negative`, `1: Neutral`, `2: Positive`.
* **Preprocessing:** Stratified Split (80/20) to ensure validation data represents all sentiment classes equally.

## 🧠 Model Architecture & Training Strategy
1.  **Tokenization:** Uses the pre-trained BERT tokenizer with truncation and padding.
2.  **Class Weights:** Before training, the script calculates the distribution of labels and assigns higher weights to minority classes (Negative/Neutral) to improve recall.
3.  **Optimization:** * **Optimizer:** AdamW (`lr=1e-5`)
    * **Scheduler:** Linear schedule with warmup (600 steps)
    * **Batch Size:** 8 per core (Effective batch size = 64 on 8-core TPU).
4.  **Training Loop:** Runs for 12 Epochs using XLA multiprocessing to sync gradients across cores.

## 📉 Results
*After training for 12 epochs, the model achieved:*

| Metric | Score |
| :--- | :--- |
| **Accuracy** | 0.9912  |
| **F1-Score (Macro)** | 0.9930  |
| **Precision** | 0.9930 |
| **Recall** | 0.9930  |

### Example Inference
**Input:** *"that's how you use power of money, not billion dollars cars. Bless him"*
**Predicted:** `Positive` (Class 2)


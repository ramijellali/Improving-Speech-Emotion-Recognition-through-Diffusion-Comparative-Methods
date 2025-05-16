# 🎤 Improving Speech Emotion Recognition through Diffusion & Comparative Methods

This project explores advanced deep learning techniques to enhance the recognition of emotions in speech, focusing on **data enrichment** and **model architecture comparison**. We implement and evaluate two approaches: a **Conditional Diffusion Model** and a **CNN + Transformer** hybrid, using two widely recognized emotional speech datasets.

---

## 🧠 Objective

Speech Emotion Recognition (SER) is crucial for building emotionally intelligent systems. However, traditional approaches suffer from:

- Sparse and noisy audio data  
- Limited expressiveness  
- Poor generalization across speakers

**Goal:** Enrich emotional speech data and improve SER model performance through advanced preprocessing and deep learning techniques.

---

## 📁 Datasets

### 🎧 EmoDB
- **Language**: German  
- **Emotions**: Neutral, Joy, Sadness, Anger, Fear, Disgust  
- **Total Samples**: 454  
- **Actors**: 10  
- **Resampled to**: 22,025 Hz

### 🎧 RAVDESS
- **Language**: English  
- **Emotions**: Neutral, Joy, Sadness, Anger, Fear, Disgust, Surprise  
- **Total Samples**: 1,056  
- **Actors**: 24  
- **Resampled to**: 22,025 Hz

---

## 🔄 Preprocessing Pipeline

1. **Resample audio** to 22,025 Hz  
2. **Adjust duration** to 10 seconds (padding)  
3. **Convert to Mel-Spectrograms** using STFT  
4. **Normalize using Z-Score**  
5. Final result: image-like spectrograms used as model input

---

## 🧪 Methods

### 🌫️ Method 1: Conditional Diffusion Model (DDPM)
- Input: Normalized Mel-spectrograms (1 × 128 × 300)
- Architecture: U-Net with temporal and emotional embeddings
- Goal: Reconstruct enriched spectrograms for better emotion modeling

### 🧠 Method 2: CNN + Transformer Hybrid
- **CNN (ResNet-50)** captures local visual patterns
- **Transformer** extracts global and temporal relationships
- Output: Predicted emotional class from processed spectrograms

---

## 📊 Results

| Method                | Accuracy | F1-Score |
|-----------------------|----------|----------|
| ResNet-50 (Fine-Tuned)| 94.36%   | 94.38%   |
| CNN + Transformer     | 72.00%   | 72.50%   |

The diffusion-based preprocessing enhanced the quality of input features, significantly benefiting the downstream classifier.

---

## 🚀 Getting Started

```bash
# Clone the repository
git clone https://github.com/ramijellali/Improving-Speech-Emotion-Recognition-through-Diffusion-Comparative-Methods.git
cd Improving-Speech-Emotion-Recognition-through-Diffusion-Comparative-Methods

# Install required packages
pip install -r requirements.txt

# Run the main script (adjust if necessary)
python main.py

# 🎥📝 Video Captioning with RNNs — Full Training Process

This document outlines a real-world RNN-based video captioning system, explaining the training dataset, input/output format, data preprocessing, loss calculation, and expected model output.

---

## 1. 🗂️ Training Dataset

A typical dataset for video captioning consists of:
- **Video files** (or extracted frame sequences)
- **Captions** describing each video

### 📦 Example Datasets:
| Dataset      | Description                                 |
|--------------|---------------------------------------------|
| MSVD         | Short YouTube clips (~2k videos, multi-caption) |
| MSR-VTT      | 10k videos with multiple captions           |
| ActivityNet Captions | Long-form video with dense descriptions |

Each video is paired with one or more ground-truth captions:
```
video_001.mp4 → ["A man is cooking in the kitchen.", "Someone is preparing food."]
video_002.mp4 → ["A dog is running in a field."]
```

---

## 2. 📥 Input Data (to the Model)

Each video is converted into a sequence of **visual features**:

- Sample video frames at a fixed rate (e.g., 1 frame per second).
- For each frame, extract visual features using a pretrained CNN (e.g., ResNet, Inception).

### Example:
```
Input video (10 frames) → [v₁, v₂, ..., v₁₀]
Each vₖ ∈ ℝⁿ, e.g., n = 2048
```

This results in a 2D input: `(num_frames, feature_dim)`

---

## 3. 📤 Output Data (Target Caption)

Each caption is tokenized into a sequence of words:

```
"A man is cooking." → ["<BOS>", "A", "man", "is", "cooking", ".", "<EOS>"]
```

Each word is mapped to an integer index using a vocabulary.

Captions are padded/truncated to a fixed maximum length, e.g., 20 tokens.

---

## 4. 🧹 Data Preprocessing

### 🖼️ Video Frame Preprocessing:
- Resize/crop frames to match CNN input size (e.g., 224×224).
- Normalize pixel values.
- Extract features from each frame using a CNN (output: 2048-D vector).
- Stack frame features to get a matrix of shape `(T, D)`.

### 📝 Caption Preprocessing:
- Lowercase text
- Remove punctuation (optional)
- Tokenize into words
- Convert words to indices using a vocabulary
- Pad to fixed length (e.g., 20)

---

## 5. 🧠 Model Architecture (RNN-Based)

### Encoder:
- Sequence of frame features: `[v₁, v₂, ..., vₜ]`
- Optionally passed through a BiLSTM to encode temporal structure.

### Decoder:
- RNN (typically LSTM) generates words one-by-one.
- Uses the previous word and encoder context at each time step.
- Output is passed through a `Dense + Softmax` layer to predict next word.

### Optional:
- Add an attention mechanism to align frames with words dynamically.

---

## 6. 🧮 Loss Calculation

The model is trained using **categorical cross-entropy loss** between predicted and ground-truth words at each time step.

### Formula:
**Loss function**  
Cross-entropy per word
```
L = - Σₜ log P(yₜ | y₁, …, yₜ₋₁, video)
```

- $begin:math:text$ y_t $end:math:text$: target word at time step $begin:math:text$ t $end:math:text$
- The loss is averaged over the time steps and batch.

---

## 7. 🏋️ Training Process

1. Extract CNN features for all video frames.
2. Convert captions into padded token sequences.
3. Feed video features into the encoder (e.g., BiLSTM).
4. Use the decoder (LSTM) to generate the caption word-by-word.
5. Compute cross-entropy loss between predicted and true words.
6. Use teacher forcing (feeding ground-truth previous word).
7. Optimize with Adam or similar optimizer over many epochs.

---

## 8. 🔚 Model Output

After training, the model:
- Takes a video (frames → CNN features)
- Generates a caption one word at a time
- Stops at the `<EOS>` token

### Example Output:
```
Input video → "A woman is slicing a tomato."
```

---

## ✅ Summary Table

| Component        | Description                                        |
|------------------|----------------------------------------------------|
| Input            | Sequence of frame features (e.g., 10 × 2048)       |
| Output           | Caption as token indices (e.g., max len = 20)      |
| Encoder          | CNN → (Optional) RNN over frame features           |
| Decoder          | LSTM + Dense + Softmax for word prediction         |
| Loss Function    | Cross-entropy between predicted and true words     |
| Output Caption   | Generated text describing the video                |

---

## 🧪 Evaluation Metrics

| Metric   | Purpose                                          |
|----------|--------------------------------------------------|
| BLEU     | Measures n-gram overlap with ground truth        |
| METEOR   | Considers synonyms, stems, word order            |
| CIDEr    | Uses TF-IDF weighting of n-grams for captioning  |
| ROUGE    | Common in summarization and captioning           |

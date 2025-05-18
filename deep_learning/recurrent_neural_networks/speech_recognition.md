## 🗣️ **What Is Speech Recognition?**

**Speech recognition** (or **Automatic Speech Recognition – ASR**) is the task of converting **spoken language** (audio) into **written text**.

### 🔍 Example:

* **Input (Audio)**: Someone says *"What's the weather today?"*
* **Output (Text)**: `"what's the weather today?"`

This process must handle:

* Accents
* Background noise
* Variable speaking speeds
* Contextual word prediction

---

## 🧠 **Why Use RNNs for Speech Recognition?**

Speech is inherently **sequential** and **time-dependent** — sounds unfold over time, and each sound (phoneme) depends on what came before and after.

**Recurrent Neural Networks (RNNs)** — especially **LSTMs** and **GRUs** — are designed to model such **temporal sequences**, making them ideal for speech recognition.

---

## 🔁 **How RNNs Work in Speech Recognition**

### 🧱 Basic Pipeline:

1. **Audio Input (Raw Sound)**

   * Audio waveform is recorded (a time-series signal).
   * Preprocessed into **spectrograms** or **MFCCs** (Mel-Frequency Cepstral Coefficients), which are 2D time–frequency representations.

2. **RNN Model**

   * The spectrogram is fed into an RNN (e.g., BiLSTM).
   * The RNN processes audio **frame by frame** (like words in a sentence).
   * The RNN learns **temporal dependencies** — which sounds lead to which letters or words.

3. **Output**

   * Model outputs either:

     * A sequence of **characters** (character-level decoding), or
     * A sequence of **words** (word-level decoding)
   * Decoding often involves **CTC (Connectionist Temporal Classification)** or attention-based decoders.

---

### 🔠 Character-Level RNN Example

Say we input the audio of: *"hello"*

The RNN might produce a character-by-character prediction:

| Time Frame | Predicted Output |
| ---------- | ---------------- |
| Frame 1    | h                |
| Frame 2    | e                |
| Frame 3    | l                |
| Frame 4    | l                |
| Frame 5    | o                |

---

## 🔣 **Connectionist Temporal Classification (CTC)**

**CTC** is often used with RNNs for ASR when **alignment between audio and labels is unknown**.

### Why?

* Audio and text sequences are **different lengths**.
* CTC allows the model to **align predictions** over time by introducing a "blank" token and collapsing repeated characters.

### Example:

Predicted sequence:
`"h", "e", "l", "l", "o", "blank"`
→ Collapsed to `"hello"`

CTC enables **end-to-end training** without needing to mark exact timing of each word or letter in the audio.

---

## 🔁 **Bidirectional RNNs in Speech Recognition**

* In **Bidirectional RNNs**, the model processes the input **forward and backward** in time.
* This helps capture **context from both past and future sounds**, improving recognition accuracy.

---

## 💡 Advanced Architectures

Modern ASR systems often build on RNNs with improvements:

* **Deep RNNs**: Stacked LSTM/GRU layers for richer representations.
* **Attention Mechanisms**: For focusing on relevant parts of the input.
* **Encoder–Decoder Models**: Especially useful in speech-to-text tasks.
* **Transformers** (replacing RNNs in some newer systems)

But RNNs are still foundational, especially for streaming or real-time ASR.

---

## 🔍 **Real-World Use Cases**

| Application                      | How RNNs Are Used                       |
| -------------------------------- | --------------------------------------- |
| **Voice Assistants**             | Siri, Alexa, Google Assistant           |
| **Live Captioning**              | Subtitles in Zoom, YouTube, Google Meet |
| **Voice Search**                 | Search by voice in browsers or apps     |
| **Call Center Transcripts**      | Convert customer calls to text          |
| **Voice Commands**               | Smart home control, in-car assistants   |
| **Speech-to-Speech Translation** | First step is speech-to-text            |

---

## 📊 Evaluation Metrics

| Metric                         | Description                                                             |
| ------------------------------ | ----------------------------------------------------------------------- |
| **WER (Word Error Rate)**      | % of words incorrectly predicted (insertions, deletions, substitutions) |
| **CER (Character Error Rate)** | Especially for character-level decoding                                 |
| **Accuracy**                   | % of correctly predicted words                                          |
| **Latency**                    | Time taken to process audio and produce text (important for real-time)  |

---

## ✅ Summary Table

| Feature                  | Description                                       |
| ------------------------ | ------------------------------------------------- |
| **Task**                 | Convert speech audio to text                      |
| **Input**                | Preprocessed audio (e.g., spectrograms)           |
| **Model**                | RNN / LSTM / GRU / BiLSTM (with CTC or attention) |
| **Output**               | Sequence of characters or words                   |
| **Key Strength of RNNs** | Captures time dependencies in audio               |
| **Used in**              | Voice assistants, transcription, accessibility    |

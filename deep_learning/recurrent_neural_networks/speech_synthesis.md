## 🗣️🔁 What is Speech Synthesis?

**Speech synthesis** is the task of converting **text input** into **natural-sounding speech output**.
It’s the technology behind:

* Virtual assistants (e.g., Siri, Alexa)
* Screen readers
* Language learning apps
* Assistive devices

---

## 🎯 Goal of RNN-based TTS

Given an input like:

> `"Hello, how are you today?"`

An RNN-based speech synthesis system should generate:

* A sequence of **acoustic features** or **waveforms** that can be vocally played back.
* Ultimately, **waveform audio** representing human-like speech.

---

## 🧠 Why Use RNNs?

Speech is a **sequential, time-dependent signal**. RNNs, particularly LSTM/GRU variants, are well-suited because they:

* Maintain temporal memory
* Model long-range dependencies in speech (prosody, rhythm, coarticulation)

---

## 🧱 System Components

### A typical RNN-based TTS system has **three major stages**:

| Stage                       | Description                                            | Output                          |
| --------------------------- | ------------------------------------------------------ | ------------------------------- |
| **1. Text Front-End**       | Converts text to phonemes/linguistic features          | e.g., `["HH", "AH", "L", "OW"]` |
| **2. Acoustic Model (RNN)** | Predicts acoustic features like spectrograms from text | e.g., Mel-spectrogram           |
| **3. Vocoder**              | Converts acoustic features to waveform                 | e.g., waveform `.wav`           |

---

## 🎧 Acoustic Modeling (with RNNs)

The **acoustic model** is the core of the synthesis process.

### Input:

* Sequence of **linguistic units** (phonemes, characters, word embeddings)
* Optional **prosody features** (e.g., pitch, stress, duration)

### Output:

* Frame-by-frame **spectrogram** features:

  * Mel-spectrogram
  * MFCCs
  * Log-magnitude STFT

---

### 🔄 RNN-Based Model Architecture:

```
Text/Phoneme Embeddings → BiLSTM/GRU Stack → Dense Layer → Spectrogram Frames
```

* **BiLSTM** layers model sequential context.
* **Dense** layer predicts acoustic features per frame.
* Optional **attention mechanism** aligns text and audio timing.

---

## 📚 Training the RNN-Based TTS Model

### 🧾 Dataset

Requires paired **text and audio**:

| Text                        | Audio File  |
| --------------------------- | ----------- |
| "Hello, how are you today?" | `hello.wav` |

* Common datasets:

  * **LJSpeech** (English, 13k sentences)
  * **Blizzard Challenge**
  * **VCTK** (multi-speaker)

---

### 🧹 Preprocessing

1. **Text Normalization**

   * Convert numbers, abbreviations:

     > `"I'm 5ft tall." → "I am five feet tall"`

2. **Text-to-Phoneme Conversion** (optional)

   * `"Hello"` → `["HH", "AH", "L", "OW"]`

3. **Audio Processing**

   * Convert to **Mel-spectrogram** or other features:

     * Frame length: 50 ms
     * Frame hop: 12.5 ms
     * Feature dim: 80–128

4. **Alignment**

   * Pair each input text with corresponding sequence of audio frames

---

## 🧮 Loss Function

* **Mean Squared Error (MSE)** or **L1 loss** between predicted and target acoustic features:

$$
\mathcal{L} = \frac{1}{T} \sum_{t=1}^{T} \| \hat{x}_t - x_t \|^2
$$

Where:

* $\hat{x}_t$: predicted spectrogram at time $t$

* $x_t$: ground-truth spectrogram frame

* Some models also use **duration loss** or **alignment loss** if using attention.

---

## 🔊 Vocoder: From Features to Waveform

The vocoder converts spectrograms into waveforms.

### Options:

| Method          | Notes                                    |
| --------------- | ---------------------------------------- |
| **Griffin-Lim** | Simple but low-quality                   |
| **WaveNet**     | RNN/Autoregressive neural vocoder        |
| **HiFi-GAN**    | Fast, high-quality, GAN-based            |
| **WaveGlow**    | Flow-based vocoder (faster than WaveNet) |

In early RNN-based systems (e.g., Tacotron 1), **WaveNet** was used as the vocoder.

---

## ⚙️ Key RNN-Based TTS Models

### 📌 Tacotron 1 (2017)

* Input: Characters
* Encoder: CBHG + BiLSTM
* Decoder: Attention + LSTM → Spectrogram
* Vocoder: WaveNet

### 📌 Deep Voice (Baidu)

* Modular pipeline (text → duration → pitch → waveform)
* Heavy use of GRUs

---

## ✅ Evaluation Metrics

| Metric                            | Description                                |
| --------------------------------- | ------------------------------------------ |
| **MOS (Mean Opinion Score)**      | Human-rated quality score (1–5)            |
| **Mel cepstral distortion (MCD)** | Difference in spectral shape               |
| **Alignment plots**               | Visualize attention between text and audio |

---

## 📈 Benefits of RNNs in TTS

* Sequence modeling of speech naturally fits RNNs
* Handles prosody and rhythm better than frame-by-frame models
* Bi-directional context (in Tacotron-like models) improves quality

---

## ❗ Limitations of RNN-Based TTS

| Issue                | Description                             |
| -------------------- | --------------------------------------- |
| **Slow inference**   | Autoregressive generation is sequential |
| **Alignment issues** | Attention sometimes fails or misaligns  |
| **Large memory use** | Especially for long sentences           |

---

## 🔄 Evolution

While early TTS systems (Tacotron, DeepVoice) used **RNNs**, modern systems like **FastSpeech 2** and **VITS** use:

* Transformer encoders
* Variational flows
* Non-autoregressive decoding
* GAN vocoders (HiFi-GAN)

---

## ✅ Summary Table

| Component     | Description                                  |
| ------------- | -------------------------------------------- |
| Input         | Text (characters, phonemes, words)           |
| Preprocessing | Text normalization, spectrogram extraction   |
| RNN Model     | BiLSTM/GRU encoder-decoder                   |
| Output        | Spectrograms (mel, STFT, MFCC)               |
| Loss          | MSE / L1 between predicted and real features |
| Vocoder       | WaveNet, WaveGlow, HiFi-GAN, etc.            |

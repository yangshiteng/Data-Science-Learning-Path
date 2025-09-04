## Example: Load & Use Pretrained BigGAN from Hugging Face

### 1. Install Required Libraries

```bash
pip install torch torchvision
pip install pytorch-pretrained-biggan
```

### 2. Load the Model and Generate Images

```python
import torch
from pytorch_pretrained_biggan import BigGAN, one_hot_from_names, truncated_noise_sample
from torchvision.utils import save_image

# Load the pretrained BigGAN model (DeepMind’s ImageNet-trained variant)
model = BigGAN.from_pretrained('biggan-deep-256')
model.eval()

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Sample a noise vector and class vector for "goldfish"
noise_vector = torch.tensor(truncated_noise_sample(truncation=0.4, batch_size=1), device=device)
class_vector = torch.tensor(one_hot_from_names(['goldfish']), device=device)

# Generate an image
with torch.no_grad():
    output = model(noise_vector, class_vector, truncation=0.4)

# Save output image
save_image(output.clamp(-1, 1), 'biggan_goldfish.png')
print("Generated image saved to biggan_goldfish.png")
```

### What’s Happening:

* We load the **BigGAN** model adapted by Hugging Face using `from_pretrained`.
* We sample a **latent noise vector** and a **class vector** (for example, `goldfish`).
* We generate an image using the model and save it locally.

This example demonstrates loading and using a GAN model from Hugging Face with just a few lines of code. The image (e.g., a goldfish) will be saved as `biggan_goldfish.png`.

---

## Additional GAN Models on Hugging Face

Beyond BigGAN, Hugging Face also hosts other GAN-based models, including:

* **`keras-io/conditional-gan`**: A Keras-based Conditional GAN for generating MNIST digits ([huggingface.co][1], [huggingface.co][2])
* **`nvidia/tts_hifigan`**: HiFi-GAN, used as an **audio vocoder** to generate speech from spectrograms ([huggingface.co][3])
* **`facebook/ic_gan`**: Instance-Conditioned GAN with a working Colab notebook to test generation ([huggingface.co][4])

You can load these models similarly—using their `from_pretrained` methods when supported or following their model card instructions.

---

## Summary Table

| Use Case                        | Model                      | Code Example Provided? |
| ------------------------------- | -------------------------- | ---------------------- |
| High-quality image generation   | BigGAN (`biggan-deep-256`) | Yes                    |
| Conditioned generation on MNIST | `keras-io/conditional-gan` | Yes (Keras)            |
| Audio waveform generation       | `tts_hifigan` (HiFi-GAN)   | Yes (NeMo toolkit)     |
| Instance-conditioned generation | `facebook/ic_gan`          | Yes (Colab)            |

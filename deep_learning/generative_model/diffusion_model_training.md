## 🧭 Overview

1. **Understand what you're training**
2. **Prepare your dataset**
3. **Build a diffusion schedule**
4. **Design your neural network (usually a U-Net)**
5. **Train the model using noise prediction**
6. **Sample images by reversing the noise process**

---

## 1. 📚 Understand the Training Objective

You’re training a model to **predict the noise** added to images at different diffusion timesteps.

> Training objective:

$$
\mathcal{L}_{\text{simple}} = \mathbb{E}_{x, t, \epsilon} \left[ \left\| \epsilon - \epsilon_\theta(x_t, t) \right\|^2 \right]
$$

Where:

* $x_t$: Noisy version of input image
* $t$: Time step
* $\epsilon$: True Gaussian noise
* $\epsilon_\theta$: Model's predicted noise

---

## 2. 🖼️ Prepare the Dataset

Use a dataset of images like CIFAR-10, MNIST, or your own collection.

```python
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor()
])

dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
```

---

## 3. 📈 Create the Diffusion Schedule

You need to define how noise is added over time.

```python
import torch

T = 1000  # number of steps
beta = torch.linspace(1e-4, 0.02, T)
alpha = 1. - beta
alpha_hat = torch.cumprod(alpha, dim=0)
```

This will help you compute noisy versions of the image:

$$
x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon
$$

---

## 4. 🧠 Build a U-Net Model (Backbone of DDPM)

A simple U-Net-style architecture is often used. You can start small:

```python
import torch.nn as nn

class SimpleUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, 3, padding=1),
        )

    def forward(self, x, t):
        return self.model(x)
```

> For serious training, use time embeddings and a full U-Net like in [🧨 Hugging Face Diffusers](https://github.com/huggingface/diffusers).

---

## 5. 🏋️‍♀️ Training Loop

Here you generate noisy images and teach your model to predict the noise.

```python
model = SimpleUNet().to("cuda")
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(10):
    for x, _ in dataloader:
        x = x.to("cuda")
        t = torch.randint(0, T, (x.size(0),), device="cuda").long()
        noise = torch.randn_like(x)

        alpha_hat_t = alpha_hat[t].view(-1, 1, 1, 1)
        noisy_x = (alpha_hat_t.sqrt() * x) + ((1 - alpha_hat_t).sqrt() * noise)

        predicted_noise = model(noisy_x, t)
        loss = ((noise - predicted_noise) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch}: loss {loss.item():.4f}")
```

---

## 6. 🖼️ Sampling: Generate Images from Noise

To generate an image, start from pure noise and denoise step-by-step using your model.

```python
@torch.no_grad()
def sample(model):
    x = torch.randn(1, 3, 32, 32).to("cuda")
    for t in reversed(range(T)):
        beta_t = beta[t]
        alpha_t = alpha[t]
        alpha_hat_t = alpha_hat[t]

        noise_pred = model(x, torch.tensor([t], device="cuda"))
        x = (1 / alpha_t.sqrt()) * (x - (beta_t / (1 - alpha_hat_t).sqrt()) * noise_pred)
        if t > 0:
            x += torch.randn_like(x) * beta_t.sqrt()
    return x
```

---

## 🔧 Optional: Use Hugging Face `diffusers`

Instead of building everything from scratch, use Hugging Face’s `diffusers`:

```bash
pip install diffusers accelerate
```

```python
from diffusers import DDPMPipeline

pipeline = DDPMPipeline.from_pretrained("google/ddpm-cifar10-32")
image = pipeline().images[0]
image.show()
```

You can even fine-tune this model on your own dataset.

---

## 📌 Summary

| Step               | What You Do                             |
| ------------------ | --------------------------------------- |
| Dataset prep       | Load and transform image data           |
| Diffusion schedule | Decide how noise is added               |
| Model              | Build U-Net to predict added noise      |
| Training           | Predict noise from noisy images         |
| Sampling           | Start with noise → denoise step-by-step |

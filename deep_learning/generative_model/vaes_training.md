## 🧠 Overview: What You’ll Do

1. **Prepare your dataset**
2. **Define the VAE architecture**
3. **Train the VAE**
4. **Visualize results** (like reconstructions or latent traversals)

---

## 📁 Step 1: Prepare Your Dataset

### ✅ Examples

* Image data (e.g. faces, handwritten digits, X-rays)
* Use a dataset from Hugging Face Datasets, Torchvision, or load your own folder

```python
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

dataset = ImageFolder("your_dataset_path", transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
```

---

## 🏗️ Step 2: Define a Simple VAE Architecture

Here’s a minimalist PyTorch VAE:

```python
import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, latent_dim=20):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*64*3, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim * 2)  # outputs both mean and logvar
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 64*64*3),
            nn.Sigmoid(),
            nn.Unflatten(1, (3, 64, 64))
        )

    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = h.chunk(2, dim=1)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar
```

---

## 🧮 Step 3: Define Loss and Train

```python
def loss_function(x_recon, x, mu, logvar):
    recon_loss = nn.functional.mse_loss(x_recon, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss

vae = VAE().to("cuda" if torch.cuda.is_available() else "cpu")
optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)

for epoch in range(20):
    for x, _ in dataloader:
        x = x.to(vae.device)
        x_recon, mu, logvar = vae(x)
        loss = loss_function(x_recon, x, mu, logvar)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}: Loss = {loss.item():.2f}")
```

---

## 🖼️ Step 4: Visualize Results

```python
import matplotlib.pyplot as plt

def show_image(img_tensor):
    img = img_tensor.detach().cpu().permute(1,2,0).numpy()
    plt.imshow(img)
    plt.axis('off')
    plt.show()

vae.eval()
with torch.no_grad():
    for x, _ in dataloader:
        x = x.to(vae.device)
        x_recon, _, _ = vae(x)
        show_image(x[0])
        show_image(x_recon[0])
        break
```

---

## 🌟 Optional: Latent Traversal / Interpolation

You can sample or interpolate between latent vectors to generate new content:

```python
z1 = torch.randn(1, 20)
z2 = torch.randn(1, 20)
for alpha in torch.linspace(0, 1, steps=5):
    z = z1 * (1 - alpha) + z2 * alpha
    image = vae.decoder(z)
    show_image(image[0])
```

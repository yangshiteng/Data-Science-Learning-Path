## 🧠 What You’re Training

You’ll train **two neural networks**:

* **Generator (G)**: learns to create data
* **Discriminator (D)**: learns to detect fake data

They play a **minimax game**:

> The generator tries to fool the discriminator, and the discriminator tries to catch it.

---

## 🧾 Step-by-Step: How to Train a GAN on Your Own Data

---

### 📦 1. Prepare Your Dataset

* Your dataset should be a **collection of similar images**, e.g. faces, flowers, shoes, artwork, etc.
* Images should be:

  * Resized to the same shape (e.g., 64x64, 128x128)
  * Stored in a folder
* **Example format:**

```
my_dataset/
├── image1.jpg
├── image2.jpg
├── ...
```

---

### 🧰 2. Install Dependencies

You'll need:

```bash
pip install torch torchvision matplotlib
```

Optional (for showing image grids):

```bash
pip install tqdm
```

---

### 🧱 3. Define the GAN Architecture (PyTorch Example)

```python
import torch
import torch.nn as nn

# Generator
class Generator(nn.Module):
    def __init__(self, z_dim=100, img_dim=784):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, img_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

# Discriminator
class Discriminator(nn.Module):
    def __init__(self, img_dim=784):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(img_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)
```

---

### 🧪 4. Load and Preprocess Your Data

```python
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.Grayscale(),  # optional
    transforms.Resize(28),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

dataset = datasets.ImageFolder(root='my_dataset', transform=transform)
loader = DataLoader(dataset, batch_size=32, shuffle=True)
```

---

### 🏋️ 5. Train the GAN

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
generator = Generator().to(device)
discriminator = Discriminator().to(device)

criterion = nn.BCELoss()
g_opt = torch.optim.Adam(generator.parameters(), lr=2e-4)
d_opt = torch.optim.Adam(discriminator.parameters(), lr=2e-4)

for epoch in range(50):
    for real, _ in loader:
        real = real.view(-1, 784).to(device)
        batch_size = real.size(0)

        # Labels
        ones = torch.ones(batch_size, 1).to(device)
        zeros = torch.zeros(batch_size, 1).to(device)

        # --- Train Discriminator ---
        noise = torch.randn(batch_size, 100).to(device)
        fake = generator(noise)

        d_loss_real = criterion(discriminator(real), ones)
        d_loss_fake = criterion(discriminator(fake.detach()), zeros)
        d_loss = d_loss_real + d_loss_fake
        d_opt.zero_grad()
        d_loss.backward()
        d_opt.step()

        # --- Train Generator ---
        output = discriminator(fake)
        g_loss = criterion(output, ones)
        g_opt.zero_grad()
        g_loss.backward()
        g_opt.step()

    print(f"Epoch {epoch+1}: D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")
```

---

### 🖼️ 6. Visualize Results

```python
import matplotlib.pyplot as plt

with torch.no_grad():
    noise = torch.randn(16, 100).to(device)
    generated = generator(noise).view(-1, 1, 28, 28).cpu()

grid = torchvision.utils.make_grid(generated, nrow=4, normalize=True)
plt.imshow(grid.permute(1, 2, 0).squeeze())
plt.axis('off')
plt.show()
```

---

## 💡 Tips

| Tip                                 | Why It Matters                                     |
| ----------------------------------- | -------------------------------------------------- |
| Normalize images to `[-1, 1]`       | GANs work better with `Tanh` output                |
| Start with simple GANs              | DCGAN, WGAN-GP, or conditional GANs can come later |
| Monitor losses & images             | Helps avoid mode collapse                          |
| Use pre-trained models if available | Faster convergence & better quality                |

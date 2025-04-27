# 🧠 **Transfer Learning (Feature Extraction) – Step-by-Step Example**

---

## 🔥 Overview:
- Use **pretrained ResNet18** (trained on ImageNet).
- **Freeze all layers** except the final classification layer.
- **Replace the final layer** to predict **2 classes** (e.g., Cats vs Dogs).
- Train only the last layer.
- Validate the results.

✅ This is **Feature Extraction** — not Fine-Tuning!

---

# 🛠️ **1. Install and Import Packages**

```bash
pip install torch torchvision
```

Now, import:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
```

---

# 🖼️ **2. Prepare a Small Dataset**

For example, use **ImageFolder** format:

Folder structure:
```
data/
  train/
    cat/
    dog/
  val/
    cat/
    dog/
```

Data transformations (resize, normalize):

```python
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],  # mean for ImageNet
                         [0.229, 0.224, 0.225])  # std for ImageNet
])

train_dataset = datasets.ImageFolder(root='./data/train', transform=transform)
val_dataset = datasets.ImageFolder(root='./data/val', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
```

✅ **ImageFolder** will automatically label classes based on folder names (cat = 0, dog = 1).

---

# 🏛️ **3. Load a Pretrained Model (ResNet18)**

```python
model = models.resnet18(pretrained=True)
```

---

# ❄️ **4. Freeze All Pretrained Layers**

```python
for param in model.parameters():
    param.requires_grad = False
```

✅ Now **no gradient** will flow through pretrained layers — **only the new layer will be trained**!

---

# 🔧 **5. Replace the Final Classification Layer**

ResNet18’s final layer is `model.fc`, originally for 1000 classes (ImageNet).

Replace it for 2 classes:

```python
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)  # 2 output classes: cat or dog
```

✅ Only `model.fc` will have trainable parameters now.

---

# 📋 **6. Define Loss Function and Optimizer**

```python
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)  # Only train fc layer
```

✅ Notice we pass `model.fc.parameters()` to the optimizer — **not** all model parameters!

---

# 🚀 **7. Train the Model**

Move model to device (GPU or CPU):

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
```

Training loop:

```python
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels)

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct.double() / len(train_loader.dataset)

    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')
```

✅ Training will be **fast** because only the **last layer** is updating!

---

# 📈 **8. Evaluate on Validation Set**

```python
model.eval()
correct = 0

with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels)

val_acc = correct.double() / len(val_loader.dataset)
print(f'Validation Accuracy: {val_acc:.4f}')
```

✅ Use `.eval()` mode and `torch.no_grad()` during evaluation to save memory and computation.

---

# 🎯 **Summary of Steps**

| Step                     | What Happened                          |
|---------------------------|----------------------------------------|
| Load pretrained model     | ResNet18 trained on ImageNet |
| Freeze layers             | Freeze all except the final fully connected layer |
| Replace output layer      | Adapt output size to match 2 classes |
| Define optimizer          | Only optimize the new layer |
| Train only new layer      | Fast and prevents overfitting |
| Evaluate                  | Measure accuracy on validation set |

---

# ✅ **Final Takeaway**

> **Transfer Learning with Feature Extraction** allows you to train powerful CNN models even on small datasets —  
> by using the knowledge already encoded in large pretrained models.

It is one of the **most effective techniques** in real-world deep learning projects!

# 📚 **Tutorial: Transfer Learning with Fine-Tuning (Step-by-Step, Python Example)**

---

# 🧠 **What is Fine-Tuning?**

In Fine-Tuning, we **do not just freeze all layers** —  
Instead, we **unfreeze some of the pretrained layers** (especially the **later layers**) and allow them to **adjust slightly** during training to better fit the **new task**.

✅ Fine-Tuning allows the model to **adapt its high-level features** to the new dataset.

---

# 🛠️ **1. Install and Import Packages**

(Same as before — if already installed, you can skip.)

```bash
pip install torch torchvision
```

Import:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
```

---

# 🖼️ **2. Prepare the Dataset**

(Same as before — resize, normalize.)

```python
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], 
                         [0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(root='./data/train', transform=transform)
val_dataset = datasets.ImageFolder(root='./data/val', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
```

---

# 🏛️ **3. Load a Pretrained Model (ResNet18)**

```python
model = models.resnet18(pretrained=True)
```

---

# 🔧 **4. Replace the Final Layer**

Because we have 2 classes:

```python
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)
```

✅ So far, **same as Feature Extraction**.

---

# 🔥 **5. Unfreeze Some Layers (Fine-Tuning)**

Now, **selectively unfreeze** layers to fine-tune!

Here, we **unfreeze only the last convolutional block** (layer4) + the final fully connected layer (fc):

```python
for name, param in model.named_parameters():
    if 'layer4' in name or 'fc' in name:
        param.requires_grad = True  # Fine-tune
    else:
        param.requires_grad = False  # Freeze earlier layers
```

✅ **layer4** contains the highest-level convolutional features — most adaptable to new tasks.

---

# 📋 **6. Define Loss Function and Optimizer**

Now the optimizer must only update the parameters we allow to train:

```python
# Only parameters that are requires_grad = True will be updated
params_to_update = [p for p in model.parameters() if p.requires_grad]

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(params_to_update, lr=0.0001)  # Smaller learning rate for fine-tuning
```

✅ Notice: **learning rate is smaller** than before (fine-tuning requires gentle updates).

---

# 🚀 **7. Train the Model**

Move model to GPU (if available):

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

✅ Here, **layer4 and fc** are adjusting their weights during training.

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

✅ As always: **model.eval()** and **no gradient tracking** during evaluation.

---

# 🎯 **Summary of Fine-Tuning Process**

| Step                     | What Happened |
|---------------------------|---------------|
| Load pretrained model     | ResNet18 from ImageNet |
| Replace output layer      | Match your new classes |
| Unfreeze last block       | Only unfreeze "layer4" and "fc" |
| Define optimizer          | Only optimize unfrozen layers |
| Train with small LR       | Fine-tune carefully |
| Validate                  | Check performance on new task |

---

# ✅ **Key Differences: Feature Extraction vs Fine-Tuning**

| Feature             | Feature Extraction | Fine-Tuning |
|---------------------|---------------------|-------------|
| Layers Trained       | Only new layers      | Last few layers + new layer |
| Speed                | Faster               | Slower |
| Accuracy Potential   | Good if tasks similar | Higher for harder tasks |
| Learning Rate        | Normal (e.g., 0.001)  | Smaller (e.g., 0.0001) |

---

# 🧠 **Final Takeaway**

> **Fine-Tuning** gives your pretrained CNN **extra flexibility** to adapt to new datasets —  
> improving accuracy, especially when the new task is **different** from the original training task!

When done properly, fine-tuning delivers **state-of-the-art** performance with **minimal effort** compared to full training from scratch.

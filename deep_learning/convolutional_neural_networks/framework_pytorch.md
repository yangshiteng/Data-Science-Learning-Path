# 📚 **Introduction to PyTorch**

---

# 🧠 **What is PyTorch?**

> **PyTorch** is an **open-source deep learning framework** developed by **Facebook's AI Research lab (FAIR)**.

✅ It provides **flexibility and control** over building, training, and experimenting with deep neural networks.  
✅ PyTorch is especially known for its **dynamic computation graph** and **Pythonic** feel, making it extremely popular among **researchers** and **developers**.

---

# 🛠️ **Why Use PyTorch?**

Before PyTorch:
- Deep learning frameworks (like original TensorFlow 1.x) had **static graphs**: build once, run many times.
- This made experimentation **harder**, **less intuitive**, and **debugging painful**.

✅ PyTorch solved this by introducing **dynamic graphs** —  
You can **define and modify** your neural network **at runtime**, just like regular Python code.

✅ **Easy to learn**, **easy to debug**, and **great for rapid prototyping**.

---

# 🚀 **Key Features of PyTorch**

| Feature                   | Description |
|----------------------------|-------------|
| **Dynamic Computation Graph** | Build networks "on the fly" during runtime |
| **Pythonic and Intuitive**    | Feels like native Python and NumPy |
| **Strong GPU Acceleration**   | Runs efficiently on GPUs |
| **Flexible Research-first Design** | Maximum control for custom models and operations |
| **Large Ecosystem**            | TorchVision, TorchText, TorchAudio, Detectron2, etc. |
| **Production Ready**          | TorchScript, TorchServe for deployment |
| **Excellent Community and Support** | Used widely in academia and industry |

---

# 🏛️ **Core Concepts in PyTorch**

| Concept          | Description |
|------------------|-------------|
| **Tensor**        | Multidimensional array (similar to NumPy ndarray) — basic data structure |
| **Autograd**      | Automatic differentiation engine for backpropagation |
| **Module**        | Base class for all models and layers (`torch.nn.Module`) |
| **Optimizer**     | Algorithms to update model weights (e.g., SGD, Adam) |
| **Loss Functions** | Define what the model is minimizing (e.g., CrossEntropyLoss) |

---

# 🧩 **Model Building Approaches in PyTorch**

| Approach            | Usage |
|----------------------|------|
| **Sequential API**   | Stack layers one after another easily |
| **Subclassing (Custom Model)** | Full control over forward pass — ideal for complex models |

✅ Beginners often start with **Sequential**.  
✅ Serious projects usually use **Subclassing (OOP-style)**.

---

# 🛠️ **Simple CNN in PyTorch – Step-by-Step Example**

---

## 1. Install PyTorch

```bash
pip install torch torchvision
```

✅ `torchvision` provides datasets and model utilities.

---

## 2. Import and Prepare Data

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# Load CIFAR-10 dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)

test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=False)
```

✅ Data is normalized and loaded using **DataLoader** — super efficient batching!

---

## 3. Build a Simple CNN Model

```python
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(6*6*64, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.max_pool2d(x, 2)
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```

✅ Notice how we **manually define the forward pass** —  
this gives PyTorch models **huge flexibility**!

---

## 4. Define Loss and Optimizer

```python
model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

✅ Standard cross-entropy loss for classification tasks.

---

## 5. Train the Model

```python
for epoch in range(10):
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")
```

✅ In PyTorch, **you manually control** the training loop —  
which makes it **very flexible** for research!

---

## 6. Evaluate the Model

```python
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")
```

✅ **torch.no_grad()** disables gradient calculation — saves memory during evaluation.

---

# 📊 **Summary of PyTorch CNN Code Steps**

| Step                     | Code Summary |
|---------------------------|--------------|
| Install and import libraries | `torch`, `torchvision` |
| Load and preprocess data   | Normalize images, use DataLoader |
| Build CNN model            | Subclass `nn.Module` |
| Define optimizer and loss  | Adam + CrossEntropyLoss |
| Train the model            | Manual training loop |
| Evaluate the model         | Accuracy on test set |

---

# 🧠 **Final Takeaway**

> **PyTorch** gives you **full flexibility**, **dynamic behavior**, and a **clean Pythonic feel**  
> for building, training, and deploying deep learning models —  
> making it the **favorite framework for researchers**, and increasingly for production systems too.

✅ TensorFlow = more structured, great for deployment.  
✅ PyTorch = more flexible, great for research and innovation.

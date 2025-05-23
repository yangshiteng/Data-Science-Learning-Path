# 📚 **Introduction to PyTorch**

---

## 🧠 **What is PyTorch?**

> **PyTorch** is an **open-source deep learning framework** developed by **Facebook's AI Research lab (FAIR)**.

✅ It provides **flexibility and control** over building, training, and experimenting with deep neural networks.  
✅ PyTorch is especially known for its **dynamic computation graph** and **Pythonic** feel, making it extremely popular among **researchers** and **developers**.

---

## 🛠️ **Why Use PyTorch?**

Before PyTorch:
- Deep learning frameworks (like original TensorFlow 1.x) had **static graphs**: build once, run many times.
- This made experimentation **harder**, **less intuitive**, and **debugging painful**.

✅ PyTorch solved this by introducing **dynamic graphs** —  
You can **define and modify** your neural network **at runtime**, just like regular Python code.

✅ **Easy to learn**, **easy to debug**, and **great for rapid prototyping**.

---

## 🚀 **Key Features of PyTorch**

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

## 🏛️ **Core Concepts in PyTorch**

| Concept          | Description |
|------------------|-------------|
| **Tensor**        | Multidimensional array (similar to NumPy ndarray) — basic data structure |
| **Autograd**      | Automatic differentiation engine for backpropagation |
| **Module**        | Base class for all models and layers (`torch.nn.Module`) |
| **Optimizer**     | Algorithms to update model weights (e.g., SGD, Adam) |
| **Loss Functions** | Define what the model is minimizing (e.g., CrossEntropyLoss) |

---

## 🧩 **Model Building Approaches in PyTorch**

| Approach            | Usage |
|----------------------|------|
| **Sequential API**   | Stack layers one after another easily |
| **Subclassing (Custom Model)** | Full control over forward pass — ideal for complex models |

✅ Beginners often start with **Sequential**.  
✅ Serious projects usually use **Subclassing (OOP-style)**.

---

## 🛠️ **Simple CNN in PyTorch – Step-by-Step Example**

---

### 1. Install PyTorch

```bash
pip install torch torchvision
```

✅ `torchvision` provides datasets and model utilities.

---

### 2. Import and Prepare Data

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

### 3. Build a Simple CNN Model

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

### 4. Define Loss and Optimizer

```python
model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

✅ Standard cross-entropy loss for classification tasks.

---

### 5. Train the Model

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

### 6. Evaluate the Model

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

## 📊 **Summary of PyTorch CNN Code Steps**

| Step                     | Code Summary |
|---------------------------|--------------|
| Install and import libraries | `torch`, `torchvision` |
| Load and preprocess data   | Normalize images, use DataLoader |
| Build CNN model            | Subclass `nn.Module` |
| Define optimizer and loss  | Adam + CrossEntropyLoss |
| Train the model            | Manual training loop |
| Evaluate the model         | Accuracy on test set |

---

## 🧠 **Final Takeaway**

> **PyTorch** gives you **full flexibility**, **dynamic behavior**, and a **clean Pythonic feel**  
> for building, training, and deploying deep learning models —  
> making it the **favorite framework for researchers**, and increasingly for production systems too.

✅ TensorFlow = more structured, great for deployment.  
✅ PyTorch = more flexible, great for research and innovation.

Of course!  
Let’s go through **saving and loading models in PyTorch** — **step-by-step**, very clearly and professionally.

---

# 📚 **Dynamic Computation Graph in PyTorch**

---

## 🧠 **What is a Computation Graph?**

In deep learning, a **computation graph** is:
- A structure that describes how **data flows** through the **operations** (layers, activations, etc.)  
- It keeps track of **how outputs are computed from inputs**, so gradients can be **automatically calculated** during **backpropagation**.

✅ In short: **Computation graph = map of operations for forward and backward pass**.

---

## 🛠️ **Static vs Dynamic Computation Graphs**

| Feature                        | Static Graph | Dynamic Graph (PyTorch) |
|---------------------------------|--------------|-------------------------|
| Define operations **before** running | ✅ Yes | ❌ No |
| Define operations **while** running | ❌ No | ✅ Yes |
| Graph is fixed after build | ✅ Yes | ❌ No |
| Graph changes based on input/data | ❌ No | ✅ Yes |
| Debugging with Python tools | ❌ Difficult | ✅ Easy |

✅ **Static**: Build once, run many times (Old TensorFlow 1.x, Caffe).  
✅ **Dynamic**: Build while running, modify anytime (PyTorch).

---

## 🔥 **Dynamic Computation Graph in PyTorch**

In **PyTorch**,  
> The computation graph is **created dynamically at runtime** — every time you call the **forward()** method.

✅ Each time you run a forward pass:
- A **new computation graph** is built **on the fly**.
- **Operations** (`+`, `*`, `relu`, `conv2d`, etc.) are **recorded step-by-step**.
- **PyTorch knows how to compute gradients automatically** based on the graph it built during this run.

---

## 🎯 **Mini Visual Concept**

Suppose you define this in forward():

```python
def forward(self, x):
    x = self.conv1(x)
    x = torch.relu(x)
    x = self.conv2(x)
    return x
```

PyTorch dynamically builds:

```
Input → Conv1 → ReLU → Conv2 → Output
```

✅ As soon as the forward is called,  
✅ The graph is built dynamically — no need to "compile" in advance!

---

## 🔥 **Dynamic Behavior Example**

You can even **change the graph based on input**:

```python
def forward(self, x):
    if x.size(0) > 10:  # If batch size > 10
        x = self.conv1(x)
    else:
        x = self.conv2(x)
    return x
```

✅ The graph will be **different** depending on the input during runtime!  
✅ This is **only possible** because PyTorch builds the graph **on the fly**!

---

## 🛠️ **How PyTorch Handles Dynamic Computation Graph**

- Every Tensor operation (like addition, matrix multiplication, convolution) is **registered** into a **dynamic graph** (called the Autograd Engine).
- When you call `.backward()`, PyTorch **traces back through the graph** and automatically computes **gradients** for you.
- After `.backward()`, the graph is **destroyed** — new graph will be built at the next forward pass.

✅ **One forward pass = one fresh graph**!

---

## 📊 **Summary of Dynamic Computation Graph in PyTorch**

| Item                  | PyTorch Dynamic Graph |
|------------------------|-----------------------|
| Graph built during     | Each forward pass runtime |
| Allows control flow    | ✅ Yes (if-else, loops, etc.) |
| Easy to debug          | ✅ Yes (standard Python tools) |
| Flexible architecture  | ✅ Easily modify models based on inputs |
| Graph lifetime         | Exists only during that forward-backward pass |

---

## 🧠 **Final Takeaway**

> **Dynamic computation graph** in PyTorch means:  
> you can **define, modify, and run your model dynamically at runtime** —  
> making it **extremely flexible**, **intuitive**, and **powerful** for research and real-world experiments!

✅ It's like **building a new graph on the fly** for every different input, every different forward call.

---

## 🚀 **Quick Simple Example**

```python
import torch
import torch.nn as nn

class DynamicNet(nn.Module):
    def __init__(self):
        super(DynamicNet, self).__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(10, 20)

    def forward(self, x):
        if x.mean() > 0:
            x = self.fc1(x)
        else:
            x = self.fc2(x)
        return x

model = DynamicNet()
input1 = torch.randn(5, 10)
output1 = model(input1)  # Will choose fc1 or fc2 based on input's mean
```

✅ Here, which **layer** the input goes through **depends on the input value** at runtime!

# 📚 **Saving and Loading Models in PyTorch**

---

## 🧠 **Two Main Saving Options in PyTorch**

| Method                        | What it Saves                                | Typical Usage |
|--------------------------------|----------------------------------------------|----------------|
| **1. Save Entire Model**       | Model architecture + weights                | Rarely recommended |
| **2. Save Only Model Weights (State Dictionary)** | Only weights/parameters | Best practice ✅ |

✅ **Saving only the model weights** is the **recommended** way in PyTorch.  
✅ This gives you **maximum flexibility** and **avoids problems** during model upgrades or architecture changes.

---

## 🚀 **1. Save Only Model Weights (Best Practice)**

---

### **Saving**

```python
# Assume model is an instance of a PyTorch model (nn.Module)
torch.save(model.state_dict(), 'model_weights.pth')
```
✅ This saves the **parameters** (weights and biases) — **not** the entire model class!

---

### **Loading**

First, you must **redefine** your model architecture exactly the same:

```python
# Define the model architecture first
model = SimpleCNN()

# Then load the saved weights
model.load_state_dict(torch.load('model_weights.pth'))

# Set to evaluation mode (important for batchnorm/dropout layers)
model.eval()
```

✅ `model.eval()` ensures the model behaves correctly during inference (e.g., turns off dropout).

---

## 🚀 **2. Save and Load the Entire Model (Alternative)**

---

### **Saving**

```python
torch.save(model, 'full_model.pth')
```

✅ This saves the **entire model class + weights** together.

---

### **Loading**

```python
# Load the full model directly
model = torch.load('full_model.pth')
model.eval()
```

✅ No need to redefine the model architecture manually.  
❗ **Warning**: This method can cause problems if your code changes (e.g., PyTorch version upgrade or custom layers).

Thus, **saving only the weights** is safer and more flexible!

---

## 📊 **Quick Summary: Save/Load in PyTorch**

| Action             | Code Example |
|--------------------|--------------|
| Save only weights  | `torch.save(model.state_dict(), 'weights.pth')` |
| Load only weights  | `model.load_state_dict(torch.load('weights.pth'))` + `model.eval()` |
| Save full model    | `torch.save(model, 'model.pth')` |
| Load full model    | `model = torch.load('model.pth')` + `model.eval()` |

---

## 🛠️ **Bonus Tip: Saving Optimizer State**

When you want to **resume training** later exactly from where you left off, you also save the **optimizer**:

### Save both model and optimizer

```python
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': current_epoch,
    'loss': current_loss,
}, 'checkpoint.pth')
```

### Load both model and optimizer

```python
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch']
loss = checkpoint['loss']

model.eval()  # or model.train() depending on usage
```

✅ This allows you to **continue training** from where you left off!

---

## 🎯 **Final Takeaway**

> In PyTorch, **saving the model’s weights (`state_dict`)** is the **best practice** —  
> because it gives you **maximum control, compatibility**, and **future-proofing** your deep learning projects.

Saving the **full model** is easy but **less safe** across upgrades and changes.

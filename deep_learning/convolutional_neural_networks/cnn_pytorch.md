# 📚 **CNN Application with PyTorch**

---

# 🚀 **We Will Build:**
- A **CNN** to **classify images** from the **CIFAR-10 dataset** (same task).
- Using **PyTorch** (clean, professional, and production-quality style).

---

# 🛠️ **Step 1: Install PyTorch**

If you haven't installed PyTorch yet:

```bash
pip install torch torchvision
```

✅ `torch` = core deep learning library  
✅ `torchvision` = datasets and image transformations

---

# 🛠️ **Step 2: Import Libraries**

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
```

✅ These are the main libraries for model, optimizer, dataset, and plotting.

---

# 🛠️ **Step 3: Load and Prepare the CIFAR-10 Dataset**

```python
# Define transformations (normalize pixel values to [-1, 1])
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # mean and std for RGB channels
])

# Load training and test sets
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                          shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                         shuffle=False)

# Class names
classes = ['plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']
```

✅ `DataLoader` automatically handles **batching** and **shuffling**.

---

# 🛠️ **Step 4: Visualize Sample Data**

(Optional but highly recommended!)

```python
import numpy as np

# Function to show an image
def imshow(img):
    img = img / 2 + 0.5  # Unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# Get some random training images
dataiter = iter(trainloader)
images, labels = next(dataiter)

# Show images
imshow(torchvision.utils.make_grid(images))
# Print labels
print(' '.join(classes[labels[j]] for j in range(8)))
```

✅ Always a good idea to **look at your dataset**!

---

# 🛠️ **Step 5: Build the CNN Model**

```python
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)  # (batch, 32, 32, 32)
        self.pool = nn.MaxPool2d(2, 2)               # (batch, 32, 16, 16)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1) # (batch, 64, 16, 16)
        self.fc1 = nn.Linear(64 * 8 * 8, 64)          # (after pooling)
        self.fc2 = nn.Linear(64, 10)                  # 10 output classes

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```

✅ This CNN:
- Has 2 convolutional layers + pooling
- Followed by 2 fully connected (dense) layers

---

# 🛠️ **Step 6: Define Loss Function and Optimizer**

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleCNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

✅ `CrossEntropyLoss` is perfect for multi-class classification.

✅ Move model to **GPU** if available (`cuda`) for faster training.

---

# 🛠️ **Step 7: Train the Model**

```python
for epoch in range(10):  # Loop over the dataset multiple times
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(trainloader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()  # Zero the parameter gradients
        outputs = model(inputs)  # Forward pass
        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Optimize

        running_loss += loss.item()

    print(f'Epoch [{epoch + 1}/10], Loss: {running_loss/len(trainloader):.4f}')
```

✅ Manual training loop gives **full control** and **flexibility** in PyTorch.

---

# 🛠️ **Step 8: Evaluate the Model**

```python
correct = 0
total = 0
model.eval()  # Set model to evaluation mode

with torch.no_grad():
    for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy on the test images: {100 * correct / total:.2f}%')
```

✅ `model.eval()` ensures layers like dropout and batchnorm behave correctly.

---

# 🛠️ **Step 9: Plot Training Loss (Optional)**

(Save `running_loss` for each epoch if you want to plot it.)

---

# 📈 **Summary of Full Steps**

| Step                  | Purpose                        |
|------------------------|--------------------------------|
| Install PyTorch         | Setup environment |
| Import libraries       | Load Torch, TorchVision, etc. |
| Load and normalize data | Prepare train/test data |
| Visualize sample images | Check the dataset |
| Build CNN model        | Create convolutional network |
| Define optimizer and loss | Setup training config |
| Train the model         | Train on dataset |
| Evaluate the model      | Test performance |

---

# 🧠 **Final Takeaway**

> **PyTorch** gives you **flexible, dynamic control** to build and train CNNs efficiently —  
> and **easily scales to complex models** as you grow your project!

✅ **Full control** = Great for research and experimentation.

✅ **Good practices** = Using GPU acceleration, clean model classes, and explicit training loops.

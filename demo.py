# ===================================================
# DEMO: Model Inversion Attack (OWASP ML03:2023)
# (Fully Self-Contained - Just Run This Cell!)
# ===================================================

# Step 1: Install & Import
!pip install torch torchvision matplotlib
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt

print("✅ All libraries installed and imported!")

# Step 2: Load Data and Train a "Victim" Model (on a single digit)
# We'll train a model specifically on the digit '3' to simulate a model trained on "confidential" data.
print("🔒 Training a 'victim' model on 'confidential' data (Digit 3s)...")

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
full_trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# Create a dataset containing ONLY the digit 3
target_digit = 3
indices = (full_trainset.targets == target_digit)
trainset_3 = torch.utils.data.Subset(full_trainset, indices.nonzero().squeeze())
train_loader = torch.utils.data.DataLoader(trainset_3, batch_size=64, shuffle=True)

# Define and train a simple model
class VictimModel(nn.Module):
    def __init__(self):
        super(VictimModel, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10) # Output for 10 classes

    def forward(self, x):
        x = x.view(x.size(0), -1) # Flatten
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

victim_model = VictimModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(victim_model.parameters(), lr=0.001)

# Train the victim model
victim_model.train()
for epoch in range(3):  # Quick training
    for data, target in train_loader:
        optimizer.zero_grad()
        output = victim_model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
print("✅ Victim model trained on digit 3s!")

# Step 3: The Model Inversion Attack
print("\n🎯 Launching Model Inversion Attack...")
print("   Goal: Steal the features of digit '3' from the model's memory.")

# We start from random noise and optimize it to be classified as the target digit with high confidence.
victim_model.eval()
input_shape = (1, 1, 28, 28)
target_class = target_digit

# Start with a random image
inverted_image = torch.randn(input_shape, requires_grad=True, device='cpu')
# We'll use an optimizer to adjust the random pixels
optimizer = optim.Adam([inverted_image], lr=0.1)
loss_fn = nn.CrossEntropyLoss()

# We want to maximize the probability for the target class
for i in range(1001):
    optimizer.zero_grad()
    output = victim_model(inverted_image)
    
    # The loss is negative confidence for the target class
    # Minimizing this loss == maximizing confidence for the target class
    loss = -output[0, target_class]
    loss.backward()
    optimizer.step()

    # Clamp pixel values to a valid range
    inverted_image.data = torch.clamp(inverted_image.data, -1, 1)
    
    if i % 200 == 0:
        confidence = torch.softmax(output, dim=1)[0, target_class].item()
        print(f"   Step {i}: Confidence for class {target_class} = {confidence:.2%}")

# Step 4: Analyze the Results
final_output = victim_model(inverted_image)
final_confidence = torch.softmax(final_output, dim=1)[0, target_class].item()
print(f"\n✅ Attack Complete! Final confidence for class '{target_class}': {final_confidence:.2%}")

# Step 5: Visualize the Inverted Image vs. a Real Image
print("\n👁️  Comparing inverted image with real training data:")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# Plot our inverted image
ax1.imshow(inverted_image.detach().squeeze(), cmap='gray', vmin=-1, vmax=1)
ax1.set_title(f'Inverted Image (Confidence: {final_confidence:.2%})')
ax1.axis('off')

# Plot a real '3' from the training set for comparison
real_image, real_label = trainset_3[0]
ax2.imshow(real_image.squeeze(), cmap='gray')
ax2.set_title(f'Real Training Image (Label: {real_label})')
ax2.axis('off')

plt.tight_layout()
plt.show()

# Step 6: Interpretation
print("\n🧠 **WHAT DOES THIS MEAN?**")
if final_confidence > 0.75:
    print("🎯 ATTACK SUCCESSFUL!")
    print(f"   The model inversion attack successfully reconstructed features of a '{target_digit}'.")
    print("   The inverted image (left) is what the model 'thinks' a perfect '3' looks like.")
    print("   This demonstrates that model parameters can leak information about their training data.")
else:
    print("⚠️  Attack partially successful. The inverted image captures some features but confidence is low.")
    print("   Try increasing the number of training epochs for the victim model or the inversion steps.")

print("\n🔒 This is ML03:2023 - an attacker can potentially steal sensitive training data just by querying a model.") 

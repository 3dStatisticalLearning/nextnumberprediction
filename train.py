#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import json

# Configuration
CONFIG = {
    "data_path": "Mappe_nagelneu_new.xlsx",
    "sequence_length": 15,  # Increased for more context
    "hidden_dim": 512,  # Increased capacity
    "num_heads": 8,
    "num_layers": 6,  # More layers for complex patterns
    "dropout": 0.1,  # Reduced dropout
    "learning_rate": 0.00005,  # Lower learning rate
    "batch_size": 16,  # Smaller batch for better gradients
    "epochs": 300,
    "patience": 30,
    "use_class_based": True,  # NEW: Treat as classification problem
}

print("Loading dataset...")
data = pd.read_excel(CONFIG["data_path"], header=None)
dataset = data.values.astype(np.int64)  # Keep as integers

print(f"Dataset shape: {dataset.shape}")
print(f"Value range: [{dataset.min()}, {dataset.max()}]")

# Get unique values for classification approach
unique_values = np.unique(dataset)
num_classes = len(unique_values)
value_to_idx = {val: idx for idx, val in enumerate(unique_values)}
idx_to_value = {idx: val for idx, val in enumerate(unique_values)}

print(f"Number of unique values: {num_classes}")
print(f"Unique values: {unique_values[:20]}...")  # Show first 20

# Save value mappings FIRST
np.save("value_mappings.npy", unique_values)
print("✓ Value mappings saved")

# Convert to class indices
dataset_indices = np.vectorize(value_to_idx.get)(dataset)

# Save statistics
stats = {
    "min": int(dataset.min()),
    "max": int(dataset.max()),
    "mean": float(dataset.mean()),
    "std": float(dataset.std()),
    "num_columns": int(dataset.shape[1]),
    "num_classes": int(num_classes),
    "unique_values": unique_values.tolist()
}

with open("model_stats.json", "w") as f:
    json.dump(stats, f)
print("✓ Statistics saved")

# Custom Dataset
class SequenceDataset(Dataset):
    def __init__(self, data, sequence_length):
        self.data = data
        self.sequence_length = sequence_length
        
    def __len__(self):
        return len(self.data) - self.sequence_length
    
    def __getitem__(self, idx):
        x = self.data[idx:idx + self.sequence_length]
        y = self.data[idx + self.sequence_length]
        return torch.LongTensor(x), torch.LongTensor(y)

# Improved Model with Classification Head
class ImprovedClassificationModel(nn.Module):
    def __init__(self, num_classes, num_positions, hidden_dim, num_heads, num_layers, dropout=0.1):
        super(ImprovedClassificationModel, self).__init__()
        
        self.num_classes = num_classes
        self.num_positions = num_positions
        self.hidden_dim = hidden_dim
        
        # Embedding layer - learn representations for each number
        self.embedding = nn.Embedding(num_classes, hidden_dim)
        
        # Position embedding
        self.position_embedding = nn.Embedding(num_positions, hidden_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Classification heads - one per position
        self.classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, num_classes)
            ) for _ in range(num_positions)
        ])
        
    def forward(self, x):
        batch_size, seq_len, num_pos = x.shape
        
        # Flatten for embedding: (batch, seq, pos) -> (batch*seq*pos)
        x_flat = x.reshape(-1)
        embeddings = self.embedding(x_flat)
        embeddings = embeddings.reshape(batch_size, seq_len, num_pos, self.hidden_dim)
        
        # Average embeddings across positions for each time step
        x_embed = embeddings.mean(dim=2)  # (batch, seq, hidden)
        
        # Add positional encoding
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        pos_embed = self.position_embedding(positions)
        x_embed = x_embed + pos_embed
        
        # Transformer
        x_transformed = self.transformer(x_embed)
        x_transformed = self.layer_norm(x_transformed)
        
        # Use last time step
        x_last = x_transformed[:, -1, :]  # (batch, hidden)
        
        # Predict each position independently
        outputs = []
        for classifier in self.classifiers:
            outputs.append(classifier(x_last))
        
        # Stack: (batch, num_positions, num_classes)
        return torch.stack(outputs, dim=1)

# Prepare data
sequence_length = CONFIG["sequence_length"]
dataset_obj = SequenceDataset(dataset_indices, sequence_length)

# Split
train_size = int(0.85 * len(dataset_obj))
val_size = len(dataset_obj) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(
    dataset_obj, [train_size, val_size]
)

train_loader = DataLoader(
    train_dataset, 
    batch_size=CONFIG["batch_size"], 
    shuffle=True,
    num_workers=0
)
val_loader = DataLoader(
    val_dataset, 
    batch_size=CONFIG["batch_size"], 
    shuffle=False,
    num_workers=0
)

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = ImprovedClassificationModel(
    num_classes=num_classes,
    num_positions=dataset.shape[1],
    hidden_dim=CONFIG["hidden_dim"],
    num_heads=CONFIG["num_heads"],
    num_layers=CONFIG["num_layers"],
    dropout=CONFIG["dropout"]
).to(device)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Loss function - CrossEntropy for classification
criterion = nn.CrossEntropyLoss()

# Optimizer with weight decay
optimizer = optim.AdamW(
    model.parameters(), 
    lr=CONFIG["learning_rate"],
    weight_decay=0.01,
    betas=(0.9, 0.999)
)

# Cosine annealing scheduler
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=20, T_mult=2, eta_min=1e-6
)

# Metrics tracking
def calculate_accuracy(outputs, targets):
    """Calculate exact match accuracy"""
    predictions = torch.argmax(outputs, dim=2)  # (batch, positions)
    correct = (predictions == targets).all(dim=1).float()
    return correct.mean().item()

def calculate_position_accuracy(outputs, targets):
    """Calculate per-position accuracy"""
    predictions = torch.argmax(outputs, dim=2)
    correct = (predictions == targets).float()
    return correct.mean().item()

# Training loop
best_val_loss = float('inf')
best_val_acc = 0.0
patience_counter = 0
train_losses = []
val_losses = []
train_accs = []
val_accs = []

print("\nStarting training...")
print("="*70)

for epoch in range(CONFIG["epochs"]):
    # Training phase
    model.train()
    train_loss = 0.0
    train_acc = 0.0
    train_pos_acc = 0.0
    
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_x)  # (batch, positions, num_classes)
        
        # Calculate loss for each position
        loss = 0
        for i in range(outputs.shape[1]):
            loss += criterion(outputs[:, i, :], batch_y[:, i])
        loss = loss / outputs.shape[1]
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        train_loss += loss.item()
        train_acc += calculate_accuracy(outputs, batch_y)
        train_pos_acc += calculate_position_accuracy(outputs, batch_y)
    
    train_loss /= len(train_loader)
    train_acc /= len(train_loader)
    train_pos_acc /= len(train_loader)
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    
    # Validation phase
    model.eval()
    val_loss = 0.0
    val_acc = 0.0
    val_pos_acc = 0.0
    
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            
            loss = 0
            for i in range(outputs.shape[1]):
                loss += criterion(outputs[:, i, :], batch_y[:, i])
            loss = loss / outputs.shape[1]
            
            val_loss += loss.item()
            val_acc += calculate_accuracy(outputs, batch_y)
            val_pos_acc += calculate_position_accuracy(outputs, batch_y)
    
    val_loss /= len(val_loader)
    val_acc /= len(val_loader)
    val_pos_acc /= len(val_loader)
    val_losses.append(val_loss)
    val_accs.append(val_acc)
    
    # Learning rate scheduling
    scheduler.step()
    
    # Print progress
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch [{epoch+1}/{CONFIG['epochs']}]")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | Pos Acc: {train_pos_acc*100:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc*100:.2f}% | Pos Acc: {val_pos_acc*100:.2f}%")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        print("-"*70)
    
    # Save best model based on validation accuracy
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_val_loss = val_loss
        patience_counter = 0
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_acc': train_acc,
            'val_acc': val_acc,
        }, "model.pth")
        print(f"✓ Model saved! Val Acc: {val_acc*100:.2f}%, Val Loss: {val_loss:.4f}")
        print("-"*70)
    else:
        patience_counter += 1
        if patience_counter >= CONFIG["patience"]:
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            break

print("\n" + "="*70)
print("Training completed!")
print(f"Best validation accuracy: {best_val_acc*100:.2f}%")
print(f"Best validation loss: {best_val_loss:.4f}")
print("="*70)

# Save training history
np.save("train_losses.npy", np.array(train_losses))
np.save("val_losses.npy", np.array(val_losses))
np.save("train_accs.npy", np.array(train_accs))
np.save("val_accs.npy", np.array(val_accs))

# Save configuration
CONFIG["best_val_acc"] = float(best_val_acc)
CONFIG["best_val_loss"] = float(best_val_loss)
with open("model_config.json", "w") as f:
    json.dump(CONFIG, f, indent=2)

print("\n✓ All files saved successfully!")


# In[ ]:





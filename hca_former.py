import pandas as pd
import numpy as np
import matplotlib.pyplot as pltimport torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, cohen_kappa_score, matthews_corrcoef, confusion_matrix
from sklearn.model_selection import StratifiedKFold

df = pd.read_csv('Ashrae_Buildings.csv')

# ✅ Load dataset (assuming df is already loaded)
hier_cols = ["Building type", "City", "Season", "Koppen climate classification", "Country"]
target_col = "Thermal sensation"

# Identify sensor feature columns
sensor_cols = df.drop(columns=[target_col] + hier_cols).columns.tolist()

# Extract features and target
X_sensors = df[sensor_cols].values
X_hier = df[hier_cols].values
y = df[target_col].values

# ✅ Encode hierarchical categorical features
label_encoders = {}
for i, col in enumerate(hier_cols):
    le = LabelEncoder()
    X_hier[:, i] = le.fit_transform(X_hier[:, i])
    label_encoders[col] = le

# ✅ Standardize sensor features
scaler = StandardScaler()
X_sensors = scaler.fit_transform(X_sensors)

# ✅ Apply SMOTE for class balancing
smote = SMOTE(sampling_strategy="auto", random_state=42)
X_sensors_resampled, y_resampled = smote.fit_resample(X_sensors, y)
X_hier_resampled = X_hier[np.random.choice(X_hier.shape[0], X_sensors_resampled.shape[0])]

# ✅ Split into training and test sets
X_sensors_train, X_sensors_test, X_hier_train, X_hier_test, y_train, y_test = train_test_split(
    X_sensors_resampled, X_hier_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)

# ✅ Dataset Class
class SensorDataset(Dataset):
    def __init__(self, X_sensors, X_hier, y):
        self.X_sensors = torch.tensor(X_sensors, dtype=torch.float32)
        self.X_hier = torch.tensor(X_hier, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_sensors[idx], self.X_hier[idx], self.y[idx]

# ✅ Create DataLoaders
train_dataset = SensorDataset(X_sensors_train, X_hier_train, y_train)
test_dataset = SensorDataset(X_sensors_test, X_hier_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

class SensorDataset(Dataset):
    def __init__(self, X_sensors, X_hier, y):
        self.X_sensors = torch.tensor(X_sensors, dtype=torch.float32)
        self.X_hier = torch.tensor(X_hier, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_sensors[idx], self.X_hier[idx], self.y[idx]



# ✅ Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super().__init__()
        self.embed_dim = embed_dim  # Save embed_dim
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-np.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)  # Shape: (1, max_len, embed_dim)

    def forward(self, x):
        """
        x: (batch_size, sequence_length, embed_dim)
        """
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # Convert (B, embed_dim) -> (B, 1, embed_dim)

        seq_len = x.shape[1]
        return x + self.pe[:, :seq_len, :].to(x.device)  # Ensure correct shape & device alignment




# ✅ Feature Tokenization
class FeatureTokenizer(nn.Module):
    def __init__(self, num_features, embed_dim):
        super().__init__()
        self.token_proj = nn.Linear(num_features, embed_dim)

    def forward(self, x):
        return self.token_proj(x)

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output, _ = self.attn(x, x, x)  # Standard self-attention
        x = x + self.dropout(attn_output)  # Residual connection
        return self.norm(x)



# ✅ Hierarchical Attention

class Attention_Hierarchical(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.mhsa = MultiHeadSelfAttention(dim, num_heads, dropout)  # Add MHSA before hierarchical attention

        self.qkv_building = nn.Linear(dim, dim * 3)
        self.qkv_city = nn.Linear(dim, dim * 3)
        self.qkv_season = nn.Linear(dim, dim * 3)
        self.qkv_climate = nn.Linear(dim, dim * 3)
        self.qkv_country = nn.Linear(dim, dim * 3)

        self.proj_building = nn.Linear(dim, dim)
        self.proj_city = nn.Linear(dim, dim)
        self.proj_season = nn.Linear(dim, dim)
        self.proj_climate = nn.Linear(dim, dim)
        self.proj_country = nn.Linear(dim, dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.mhsa(x)  # Apply MHSA first

        B, H, C = x.shape
        x_building = x[:, 0, :].unsqueeze(1)
        x_city = x[:, 1, :].unsqueeze(1)
        x_season = x[:, 2, :].unsqueeze(1)
        x_climate = x[:, 3, :].unsqueeze(1)
        x_country = x[:, 4, :].unsqueeze(1)

        def apply_attention(x_hier, qkv_layer, proj_layer):
            qkv = qkv_layer(x_hier)
            q, k, v = qkv.chunk(3, dim=-1)
            attn = (q @ k.transpose(-2, -1)) / np.sqrt(q.shape[-1])
            attn = attn.softmax(dim=-1)
            attn = self.dropout(attn)
            x_hier = attn @ v
            return proj_layer(x_hier)

        x_building = apply_attention(x_building, self.qkv_building, self.proj_building)
        x_city = apply_attention(x_city, self.qkv_city, self.proj_city)
        x_season = apply_attention(x_season, self.qkv_season, self.proj_season)
        x_climate = apply_attention(x_climate, self.qkv_climate, self.proj_climate)
        x_country = apply_attention(x_country, self.qkv_country, self.proj_country)

        x_city += x_building
        x_season += x_city
        x_climate += x_season
        x_country += x_climate

        x = torch.cat([x_building, x_city, x_season, x_climate, x_country], dim=1)
        return self.dropout(x)

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.mhsa = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.hta = Attention_Hierarchical(embed_dim, num_heads, dropout)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """
        x: (batch_size, sequence_length, embed_dim)
        """
        original_x = x  # Store the original tensor before applying MHSA

        x = self.mhsa(x)  # Apply Multi-Head Self-Attention


        if x.shape[1] > 5:
            x = x[:, 1:, :]

        x = self.hta(self.norm1(x))  # Hierarchical Attention


        if original_x.shape[1] > x.shape[1]:
            cls_token = original_x[:, 0, :].unsqueeze(1)
            x = torch.cat([cls_token, x], dim=1)

        x = x + self.mlp(self.norm2(x))  # Apply MLP (Feedforward network)
        return x



# ✅ Hierarchical Transformer Model
class HierarchicalTransformerWithMHSA(nn.Module):
    def __init__(self, num_sensors, num_classes, num_heads=8, depth=6, embed_dim=128, dropout=0.1):
        super().__init__()
        self.feature_tokenizer = nn.Linear(num_sensors, embed_dim)

        self.hier_embeddings = nn.ModuleList([
            nn.Embedding(len(np.unique(X_hier[:, i])), embed_dim) for i in range(X_hier.shape[1])
        ])

        self.blocks = nn.ModuleList([TransformerBlock(embed_dim, num_heads, dropout=dropout) for _ in range(depth)])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x_sensors, x_hier):
        x = self.feature_tokenizer(x_sensors)

        for i in range(x_hier.shape[1]):
            x_hier[:, i] = torch.clamp(x_hier[:, i], min=0, max=self.hier_embeddings[i].num_embeddings - 1)

        hier_embeds = [emb(x_hier[:, i]) for i, emb in enumerate(self.hier_embeddings)]
        hier_embeds = torch.stack(hier_embeds, dim=1)

        x = torch.cat([x.unsqueeze(1), hier_embeds], dim=1)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x.mean(dim=1))
        return self.head(x)


device = "cuda" if torch.cuda.is_available() else "cpu"


# Define K for K-Fold Cross-Validation
K = 10

# Define device (CUDA or CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Define the number of classes
num_classes = len(np.unique(y))

# Store results
fold_results = []

# Initialize StratifiedKFold
kf = StratifiedKFold(n_splits=K, shuffle=True, random_state=42)

# Convert data to tensors
X_sensors_tensor = torch.tensor(X_sensors, dtype=torch.float32)
X_hier_tensor = torch.tensor(X_hier, dtype=torch.long)
y_tensor = torch.tensor(y, dtype=torch.long)

# K-Fold Training Loop
for fold, (train_idx, val_idx) in enumerate(kf.split(X_sensors, y)):
    print(f"\n=== Fold {fold+1}/{K} ===")

    # Create Train and Validation Sets
    X_sensors_train, X_sensors_val = X_sensors_tensor[train_idx], X_sensors_tensor[val_idx]
    X_hier_train, X_hier_val = X_hier_tensor[train_idx], X_hier_tensor[val_idx]
    y_train, y_val = y_tensor[train_idx], y_tensor[val_idx]

    # Create PyTorch Dataset & DataLoader
    train_dataset = SensorDataset(X_sensors_train, X_hier_train, y_train)
    val_dataset = SensorDataset(X_sensors_val, X_hier_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

    # Initialize the Model
    model = HierarchicalTransformerWithMHSA(
        num_sensors=len(sensor_cols),
        num_classes=num_classes,
        num_heads=8,
        depth=6,
        embed_dim=128,
        dropout=0.1
    ).to(device)

    # Define Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    # Train the model
    num_epochs = 100  # Set number of training epochs per fold
    best_val_loss = float("inf")
    patience = 5  # Early stopping patience
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        total_loss, correct, total = 0, 0, 0
        model.train()

        for sensors, hier, labels in train_loader:
            sensors, hier, labels = sensors.to(device), hier.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(sensors, hier)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        train_loss = total_loss / len(train_loader)


        # Validate Model
        model.eval()
        val_correct, val_total, val_loss = 0, 0, 0
        with torch.no_grad():
            for sensors, hier, labels in val_loader:
                sensors, hier, labels = sensors.to(device), hier.to(device), labels.to(device)
                outputs = model(sensors, hier)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total
        val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch+1}/{num_epochs} | Train Acc: {train_acc:.4f} | Train Loss: {train_loss:.4f} | Validation Loss: {val_loss:.4f} | Validation Acc: {val_acc:.4f}")


        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"⏳ Early Stopping Triggered at Epoch {epoch+1}. Restoring Best Model...")
            model.load_state_dict(best_model_wts)
            break

    # Evaluate Model on Validation Set
    def evaluate_model(model, val_loader, device, num_classes):
        model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for sensors, hier, labels in val_loader:
                sensors, hier, labels = sensors.to(device), hier.to(device), labels.to(device)
                outputs = model(sensors, hier)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="weighted")
        kappa = cohen_kappa_score(all_labels, all_preds)
        mcc = matthews_corrcoef(all_labels, all_preds)

        print(f"\n=== Fold {fold+1} Results ===")
        print(f"✅ Accuracy: {accuracy:.4f}")
        print(f"✅ Precision: {precision:.4f}")
        print(f"✅ Recall: {recall:.4f}")
        print(f"✅ F1-Score: {f1:.4f}")
        print(f"✅ Cohen’s Kappa: {kappa:.4f}")
        print(f"✅ MCC: {mcc:.4f}")

        return accuracy, precision, recall, f1, kappa, mcc

    # Run evaluation on this fold
    fold_results.append(evaluate_model(model, val_loader, device, num_classes))

# Compute the mean performance across folds
mean_results = np.mean(fold_results, axis=0)
print("\n=== Final K-Fold Results ===")
print(f"✅ Mean Accuracy: {mean_results[0]:.4f}")
print(f"✅ Mean Precision: {mean_results[1]:.4f}")
print(f"✅ Mean Recall: {mean_results[2]:.4f}")
print(f"✅ Mean F1-Score: {mean_results[3]:.4f}")
print(f"✅ Mean Cohen’s Kappa: {mean_results[4]:.4f}")
print(f"✅ Mean MCC: {mean_results[5]:.4f}")

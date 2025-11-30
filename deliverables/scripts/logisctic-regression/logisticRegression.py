"""
Aiden Cox
CS-6840-01
Assistant Professor Dr. Wen Zhang
11/21/2025
"""

# libraries
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import numpy as np
import dataframe_image as dfi
import seaborn as sns 
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D


# model Def
class LogisticRegression(nn.Module):
    def __init__(self, input_size):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, 1)
    
    def forward(self, x):
        return torch.sigmoid(self.linear(x))

# training Function set 
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=100, patience=10):
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        # training
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # early stopping
        # if no improvement in val loss for 'patience' epochs, stop training
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    return train_losses, val_losses

# evaluation Function
def evaluate_model(model, test_loader):
    model.eval()
    y_prediction = [] # predicted labels
    y_true = [] # true labels
    y_probability = [] # predicted probabilities
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            
            y_prediction.extend((outputs > 0.5).numpy().flatten())
            y_probability.extend(outputs.numpy().flatten())
            y_true.extend(y_batch.numpy().flatten())
    
    y_prediction = np.array(y_prediction)
    y_true = np.array(y_true)
    y_probability = np.array(y_probability)
    
    acc = accuracy_score(y_true, y_prediction)
    prec = precision_score(y_true, y_prediction, zero_division=0)
    rec = recall_score(y_true, y_prediction, zero_division=0)
    f1 = f1_score(y_true, y_prediction, zero_division=0)
    cm = confusion_matrix(y_true, y_prediction)
    
    # Calculate misclassifications
    misclassified = y_true != y_prediction
    num_misclassified = np.sum(misclassified)
    num_false_positives = np.sum((y_prediction == 1) & (y_true == 0))
    num_false_negatives = np.sum((y_prediction == 0) & (y_true == 1))
    
    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'confusion_matrix': cm,
        'misclassified': misclassified,
        'num_misclassified': num_misclassified,
        'num_false_positives': num_false_positives,
        'num_false_negatives': num_false_negatives,
        'y_pred': y_prediction.astype(int),
        'y_true': y_true.astype(int)
    }

# paths of where to find and store data
asset_path = Path(__file__).parent.parent.parent.parent / 'assets' / 'logistic-regression'
data_path = Path(__file__).parent.parent.parent.parent / 'dataset-harmful-algal-bloom(HAB)' / 'HAB_Artificial_GAN_Dataset.csv'

asset_path.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(data_path)

# separate features and target 
# this is for binary classification of HAB presence

X = df.iloc[:, :-1].values
y = df['HAB_Present'].values.reshape(-1, 1)

# standardize features
# standardizing features to have mean=0 and std=1
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# defining train/test splits to experiment with
splits = {'20:80': 0.2, '40:60': 0.4, '80:20': 0.8}
results_summary = []

for split_name, train_size in splits.items():
    print(f"\n{'='*60}")
    print(f"Experiment: {split_name} (Train:Test)")
    print(f"{'='*60}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=(1-train_size)
    )
    
    # Further split training into train/val (80/20 of training data)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2
    )
    
    # convert to pytorch tensors
    # pytorch tensors are used to store data for model training
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.FloatTensor(y_val)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.FloatTensor(y_test)
    
    # create dataLoaders
    # dataloaders are used to load data in batches
    train_dataset = TensorDataset(X_train_t, y_train_t)
    val_dataset = TensorDataset(X_val_t, y_val_t)
    test_dataset = TensorDataset(X_test_t, y_test_t)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # init model and loss 
    # LEARNING RATE CHANGE HERE
    input_size = X_scaled.shape[1]
    model = LogisticRegression(input_size)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    
    # train model
    # CHANGE EPOCHS AND PATIENCE HERE
    train_losses, val_losses = train_model(model, train_loader, val_loader, criterion, optimizer, epochs=200, patience=15)
    
    # evaluations on test set
    results = evaluate_model(model, test_loader)
    
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1-Score: {results['f1']:.4f}")
    print(f"Misclassifications: {results['num_misclassified']}")
    print(f"  - False Positives: {results['num_false_positives']}")
    print(f"  - False Negatives: {results['num_false_negatives']}")   
 
    
    results_summary.append({
        'Split': split_name,
        'Accuracy': results['accuracy'],
        'Precision': results['precision'],
        'Recall': results['recall'],
        'F1': results['f1'],
    })
    
    # plot learning curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Learning Curves - {split_name} Split')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{asset_path}/learning_curves_{split_name.replace(":", "_")}.png', dpi=300, bbox_inches='tight')
    plt.close()

    
    # plotting 3D PCA visualization with misclassifications like Rose
    pca_3d = PCA(n_components=3)
    X_test_pca = pca_3d.fit_transform(X_test)
    
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    #plot correctly classified points
    correct = ~results['misclassified']
    ax.scatter(X_test_pca[correct, 0], X_test_pca[correct, 1], X_test_pca[correct, 2],
               c='purple', marker='o', s=50, label='Correctly Classified', alpha=0.6)
    
    # plot false positives (red X)
    false_pos = (results['misclassified']) & (results['y_pred'] == 1)
    ax.scatter(X_test_pca[false_pos, 0], X_test_pca[false_pos, 1], X_test_pca[false_pos, 2],
               c='red', marker='x', s=200, label='False Positive', linewidth=2)
    
    # plot false negatives (red circle)
    false_neg = (results['misclassified']) & (results['y_pred'] == 0)
    ax.scatter(X_test_pca[false_neg, 0], X_test_pca[false_neg, 1], X_test_pca[false_neg, 2],
               c='red', marker='o', s=200, label='False Negative', alpha=0.7)
    
    ax.set_xlabel('PC1', labelpad=10)
    ax.set_ylabel('PC2', labelpad=10)
    ax.set_zlabel('PC3', labelpad=10)
    ax.set_title(f'3D PCA of Test Set with Misclassifications - {split_name} Split')
    ax.legend()
    plt.tight_layout()  # ADD THIS LINE
    plt.savefig(f'{asset_path}/pca_misclassifications_{split_name.replace(":", "_")}.png', 
                dpi=300, bbox_inches='tight', pad_inches=0.5)  # ADD pad_inches=0.5
    plt.close()
    
    # plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(results['confusion_matrix'], annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['No HAB', 'HAB'], yticklabels=['No HAB', 'HAB'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title(f'Confusion Matrix - {split_name} Split')
    plt.savefig(f'{asset_path}/confusion_matrix_{split_name.replace(":", "_")}.png', dpi=300, bbox_inches='tight')
    plt.close()

# summary table
df_summary = pd.DataFrame(results_summary)
print("\n" + "="*60)
print("Summary of All Experiments:")
print(df_summary.to_string(index=False))
dfi.export(df_summary, filename=f'{asset_path}/results_summary.png', dpi=300)
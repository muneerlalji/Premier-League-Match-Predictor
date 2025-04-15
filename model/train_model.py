import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from data_processor import load_and_preprocess_data, create_data_loaders
import os
from sklearn.utils.class_weight import compute_class_weight

class MatchPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MatchPredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc4 = nn.Linear(hidden_size // 2, num_classes)
        self.dropout = nn.Dropout(0.3)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size // 2)
        
    def forward(self, x):
        output = self.fc1(x)
        output = self.bn1(output)
        output = self.relu(output)
        output = self.dropout(output)
        
        output = self.fc2(output)
        output = self.bn2(output)
        output = self.relu(output)
        output = self.dropout(output)
        
        output = self.fc3(output)
        output = self.bn3(output)
        output = self.relu(output)
        output = self.dropout(output)
        
        output = self.fc4(output)
        return output

def train_model(model, train_loader, criterion, optimizer, device, num_epochs, patience=10):
    model.train()
    loss_history = []
    best_loss = float('inf')
    no_improve = 0
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total
        loss_history.append(epoch_loss)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')
        
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            no_improve = 0
        else:
            no_improve += 1
            
        if no_improve >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    return loss_history

def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    accuracy = correct / total
    
    from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, average=None, zero_division=0
    )
    
    class_metrics = {
        'home_win': {'precision': precision[0], 'recall': recall[0], 'f1': f1[0]},
        'draw': {'precision': precision[1], 'recall': recall[1], 'f1': f1[1]},
        'away_win': {'precision': precision[2], 'recall': recall[2], 'f1': f1[2]}
    }
    
    print("\nClass metrics:")
    for class_name, metrics in class_metrics.items():
        print(f"{class_name}: Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, F1={metrics['f1']:.4f}")
    
    cm = confusion_matrix(all_targets, all_preds)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    classes = ['Home Win', 'Draw', 'Away Win']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('model/confusion_matrix.png')
    
    return accuracy

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                            'model/dataset/Premier League-Matches-1993-2023.csv')
    train_dataset, test_dataset, scaler = load_and_preprocess_data(data_path)
    train_loader, test_loader = create_data_loaders(train_dataset, test_dataset)
    
    input_size = train_dataset[0][0].size(0)
    hidden_size = 128
    num_classes = 3
    
    model = MatchPredictor(input_size, hidden_size, num_classes).to(device)
    
    all_labels = [target.item() for _, target in train_dataset]
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(all_labels),
        y=all_labels
    )
    print(f"Class weights: {class_weights}")
    
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    
    # Use weighted cross entropy loss
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    num_epochs = 200
    loss_history = train_model(model, train_loader, criterion, optimizer, device, num_epochs, patience=15)
    
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('model/training_loss.png')
    
    accuracy = evaluate_model(model, test_loader, device)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler': scaler,
    }, 'model/saved_model/premier_league_model.pth')
    
    print(f"Model accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()

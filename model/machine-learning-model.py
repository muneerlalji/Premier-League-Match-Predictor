import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle
import os

# 1. Data Preparation
class FootballDataset(Dataset):
    def __init__(self, X, y):
        # Convert sparse matrix to dense if necessary
        if hasattr(X, "toarray"):
            X = X.toarray()
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def prepare_data(df):
    # Handle missing values
    df = df.fillna(0)
    
    # Convert date to numeric features
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.month
    df['day_of_year'] = df['date'].dt.dayofyear
    
    # Create binary feature for home/away
    df['is_home'] = df['venue'] == 'Home'
    df['is_home'] = df['is_home'].astype(int)
    
    # Encode the result
    result_encoder = LabelEncoder()
    df['result_encoded'] = result_encoder.fit_transform(df['result'])
    
    # Encode teams
    team_encoder = LabelEncoder()
    df['team_encoded'] = team_encoder.fit_transform(df['team'])
    
    opponent_encoder = LabelEncoder()
    df['opponent_encoded'] = opponent_encoder.fit_transform(df['opponent'])
    
    # Select features for the model
    cat_features = ['team_encoded', 'opponent_encoded']
    num_features = ['is_home', 'month', 'day_of_year', 'poss', 'xg', 'xga', 'sh', 'sot', 'dist']
    
    # Filter out features that don't exist in the dataset
    available_num_features = [f for f in num_features if f in df.columns]
    print(f"Available numerical features: {available_num_features}")
    
    # Create a preprocessor that handles both numerical and categorical features
    # For older scikit-learn versions, OneHotEncoder doesn't have sparse parameter
    try:
        # Try new version syntax
        cat_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
    except TypeError:
        # Fall back to old version syntax
        cat_encoder = OneHotEncoder(handle_unknown='ignore')
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), available_num_features),
            ('cat', cat_encoder, cat_features)
        ])
    
    # Prepare X and y
    X = preprocessor.fit_transform(df[available_num_features + cat_features])
    y = df['result_encoded'].values
    
    return X, y, result_encoder.classes_, team_encoder, opponent_encoder, preprocessor, available_num_features

# 2. Model Architecture
class MatchPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MatchPredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn2 = nn.BatchNorm1d(hidden_size // 2)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(hidden_size // 2, output_size)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        if x.shape[0] > 1:  # BatchNorm needs more than 1 sample
            x = self.bn1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu(x)
        if x.shape[0] > 1:
            x = self.bn2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

# 3. Training Function
def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=100):
    model.to(device)
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        epoch_train_loss = running_loss / len(train_loader)
        train_losses.append(epoch_train_loss)
        
        # Validation
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        epoch_val_loss = running_loss / len(val_loader)
        epoch_val_accuracy = correct / total
        
        val_losses.append(epoch_val_loss)
        val_accuracies.append(epoch_val_accuracy)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, Val Accuracy: {epoch_val_accuracy:.4f}')
    
    return train_losses, val_losses, val_accuracies

# 4. Function to make predictions
def predict_match(model, features, result_classes, device):
    model.eval()
    with torch.no_grad():
        # Convert sparse matrix to dense if necessary
        if hasattr(features, "toarray"):
            features = features.toarray()
        features_tensor = torch.tensor(features, dtype=torch.float32).to(device)
        outputs = model(features_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predicted_idx = torch.argmax(probabilities, dim=1).item()
        predicted_result = result_classes[predicted_idx]
        
        return predicted_result, probabilities.cpu().numpy()[0]

# 5. Function to prepare prediction data
def prepare_prediction_data(team, opponent, is_home, match_date, match_stats, team_encoder, opponent_encoder, preprocessor, available_num_features):
    """
    Prepare prediction data for a future match.
    
    Args:
        team: Team name
        opponent: Opponent team name
        is_home: Whether the match is at home (1) or away (0)
        match_date: Date of the match (YYYY-MM-DD)
        match_stats: Dictionary with expected stats (poss, xg, xga, sh, sot, dist)
        team_encoder: Encoder for team names
        opponent_encoder: Encoder for opponent names
        preprocessor: The column transformer used during training
        available_num_features: List of numerical features used in training
    """
    # Convert date
    date = pd.to_datetime(match_date)
    month = date.month
    day_of_year = date.dayofyear
    
    # Encode team and opponent
    try:
        team_encoded = team_encoder.transform([team])[0]
        opponent_encoded = opponent_encoder.transform([opponent])[0]
    except:
        raise ValueError(f"Team '{team}' or opponent '{opponent}' not found in training data")
    
    # Create feature row
    data = {
        'team_encoded': team_encoded,
        'opponent_encoded': opponent_encoded
    }
    
    # Add available numerical features
    if 'is_home' in available_num_features:
        data['is_home'] = is_home
    if 'month' in available_num_features:
        data['month'] = month
    if 'day_of_year' in available_num_features:
        data['day_of_year'] = day_of_year
    
    # Add stats if they are in available features
    for stat in ['poss', 'xg', 'xga', 'sh', 'sot', 'dist']:
        if stat in available_num_features:
            data[stat] = match_stats.get(stat, 0)
    
    # Create a DataFrame with one row
    df = pd.DataFrame([data])
    
    # Transform features using the preprocessor
    X = preprocessor.transform(df)
    
    return X

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    print("Loading Premier League data...")
    try:
        data = pd.read_csv('model/dataset/FootballMatches.csv')  # Assuming the data is saved as CSV
    except FileNotFoundError:
        csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
        if csv_files:
            print(f"Premier League data not found, using {csv_files[0]} instead")
            data = pd.read_csv(csv_files[0])
        else:
            raise FileNotFoundError("No CSV files found in the current directory")
            
    print("Data loaded successfully with shape:", data.shape)
    
    # Prepare the data
    X, y, result_classes, team_encoder, opponent_encoder, preprocessor, available_num_features = prepare_data(data)
    print(f"Features shape: {X.shape}, Target shape: {y.shape}")
    print(f"Result classes: {result_classes}")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Create datasets and dataloaders
    train_dataset = FootballDataset(X_train, y_train)
    test_dataset = FootballDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Initialize the model
    input_size = X.shape[1]
    hidden_size = 128
    output_size = len(result_classes)
    
    model = MatchPredictor(input_size, hidden_size, output_size)
    print(model)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # Train the model
    print("Starting training...")
    train_model(
        model, train_loader, test_loader, criterion, optimizer, device, num_epochs=100
    )
    
    # Evaluate the model
    model.eval()
    correct = 0
    total = 0
    
    print("\nEvaluating on test set...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    test_accuracy = correct / total
    print(f'Test Accuracy: {test_accuracy:.4f}')
    
    # Save the model
    torch.save(model.state_dict(), 'model/saved_model/premier_league_predictor.pth')
    
    # Save the preprocessor, encoders, and result classes
    with open('model/saved_model/model_components.pkl', 'wb') as f:
        pickle.dump({
            'team_encoder': team_encoder,
            'opponent_encoder': opponent_encoder,
            'preprocessor': preprocessor,
            'result_classes': result_classes,
            'available_num_features': available_num_features,
            'input_size': input_size,
            'hidden_size': hidden_size,
            'output_size': output_size
        }, f)
    
    print("\nModel and components saved in 'model/saved_model' directory")


if __name__ == "__main__":
    main()
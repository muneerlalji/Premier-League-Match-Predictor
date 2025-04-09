import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
import os
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

class FootballDataset(Dataset):
    def __init__(self, X, y):
        if hasattr(X, "toarray"):
            X = X.toarray()
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def create_team_mapping(data):
    """Create unique ID mapping for teams"""
    teams = sorted(set(data['team'].unique()) | set(data['opponent'].unique()))
    team_to_id = {team: idx for idx, team in enumerate(teams)}
    id_to_team = {idx: team for idx, team in enumerate(teams)}
    return team_to_id, id_to_team

def calculate_team_stats(data, team_to_id):
    """
    Calculate historical team statistics that would be known before a match.
    This includes:
    - Form (last 5 games)
    - Home/Away form
    - Average goals scored and conceded
    - League position
    - Head-to-head record
    """
    # Create a dictionary to store team stats with date-based snapshots
    team_stats = {}
    for team_id in team_to_id.values():
        team_stats[team_id] = {}
    
    # Sort data by date
    data['date'] = pd.to_datetime(data['date'])
    sorted_data = data.sort_values('date')
    
    # Process each match to build up historical statistics
    for _, match in sorted_data.iterrows():
        date = match['date']
        team = match['team']
        opponent = match['opponent']
        result = match['result']
        venue = match['venue']
        team_id = team_to_id[team]
        opponent_id = team_to_id[opponent]
        
        # Create empty stats object if this is the first match on this date
        if date not in team_stats[team_id]:
            team_stats[team_id][date] = {
                'played': 0,
                'wins': 0,
                'draws': 0,
                'losses': 0,
                'goals_for': 0,
                'goals_against': 0,
                'home_wins': 0,
                'home_draws': 0,
                'home_losses': 0,
                'away_wins': 0,
                'away_draws': 0,
                'away_losses': 0,
                'points': 0,
                'form': [],  # List of last 5 results (W=3, D=1, L=0)
                'home_form': [],  # List of last 5 home results
                'away_form': [],  # List of last 5 away results
                'matches_against': {}  # Head-to-head record against each opponent
            }
        
        if date not in team_stats[opponent_id]:
            team_stats[opponent_id][date] = {
                'played': 0,
                'wins': 0,
                'draws': 0,
                'losses': 0,
                'goals_for': 0,
                'goals_against': 0,
                'home_wins': 0,
                'home_draws': 0,
                'home_losses': 0,
                'away_wins': 0,
                'away_draws': 0,
                'away_losses': 0,
                'points': 0,
                'form': [],  # List of last 5 results (W=3, D=1, L=0)
                'home_form': [],  # List of last 5 home results
                'away_form': [],  # List of last 5 away results
                'matches_against': {}  # Head-to-head record against each opponent
            }
        
        # Initialize head-to-head records if they don't exist
        if opponent_id not in team_stats[team_id][date]['matches_against']:
            team_stats[team_id][date]['matches_against'][opponent_id] = {
                'played': 0, 'wins': 0, 'draws': 0, 'losses': 0
            }
        
        if team_id not in team_stats[opponent_id][date]['matches_against']:
            team_stats[opponent_id][date]['matches_against'][team_id] = {
                'played': 0, 'wins': 0, 'draws': 0, 'losses': 0
            }
    
    # Now iterate through matches again to compute stats before each match
    for idx, match in sorted_data.iterrows():
        date = match['date']
        team = match['team']
        opponent = match['opponent']
        team_id = team_to_id[team]
        opponent_id = team_to_id[opponent]
        
        # Find the previous date in this team's stats
        prev_dates = [d for d in team_stats[team_id].keys() if d < date]
        if prev_dates:
            prev_date = max(prev_dates)
            # Copy stats from the previous date as the baseline
            team_stats[team_id][date] = team_stats[team_id][prev_date].copy()
            # Make a deep copy of the nested matches_against dictionary
            team_stats[team_id][date]['matches_against'] = {
                k: v.copy() for k, v in team_stats[team_id][prev_date]['matches_against'].items()
            }
            
            # Do the same for opponent
            opp_prev_dates = [d for d in team_stats[opponent_id].keys() if d < date]
            if opp_prev_dates:
                opp_prev_date = max(opp_prev_dates)
                team_stats[opponent_id][date] = team_stats[opponent_id][opp_prev_date].copy()
                # Make a deep copy of the nested matches_against dictionary
                team_stats[opponent_id][date]['matches_against'] = {
                    k: v.copy() for k, v in team_stats[opponent_id][opp_prev_date]['matches_against'].items()
                }
    
    # Now iterate again to compute match results and update stats for future matches
    for idx, match in sorted_data.iterrows():
        date = match['date']
        team = match['team']
        opponent = match['opponent']
        result = match['result']
        venue = match['venue']
        gf = match['gf'] if 'gf' in match and not pd.isna(match['gf']) else 0
        ga = match['ga'] if 'ga' in match and not pd.isna(match['ga']) else 0
        team_id = team_to_id[team]
        opponent_id = team_to_id[opponent]
        
        # Get the next match date (where we'll store the updated stats)
        next_match_dates = [d for d in team_stats[team_id].keys() if d > date]
        opponent_next_dates = [d for d in team_stats[opponent_id].keys() if d > date]
        
        # Update based on the result
        if result == 'W':
            # Update team stats
            for next_date in next_match_dates:
                team_stats[team_id][next_date]['played'] += 1
                team_stats[team_id][next_date]['wins'] += 1
                team_stats[team_id][next_date]['points'] += 3
                team_stats[team_id][next_date]['goals_for'] += gf
                team_stats[team_id][next_date]['goals_against'] += ga
                team_stats[team_id][next_date]['form'] = (team_stats[team_id][next_date]['form'] + [3])[-5:]
                
                if venue == 'Home':
                    team_stats[team_id][next_date]['home_wins'] += 1
                    team_stats[team_id][next_date]['home_form'] = (team_stats[team_id][next_date]['home_form'] + [3])[-5:]
                else:
                    team_stats[team_id][next_date]['away_wins'] += 1
                    team_stats[team_id][next_date]['away_form'] = (team_stats[team_id][next_date]['away_form'] + [3])[-5:]
                
                # Update head-to-head - ensure the entry exists first
                if opponent_id not in team_stats[team_id][next_date]['matches_against']:
                    team_stats[team_id][next_date]['matches_against'][opponent_id] = {
                        'played': 0, 'wins': 0, 'draws': 0, 'losses': 0
                    }
                team_stats[team_id][next_date]['matches_against'][opponent_id]['played'] += 1
                team_stats[team_id][next_date]['matches_against'][opponent_id]['wins'] += 1
            
            # Update opponent stats
            for next_date in opponent_next_dates:
                team_stats[opponent_id][next_date]['played'] += 1
                team_stats[opponent_id][next_date]['losses'] += 1
                team_stats[opponent_id][next_date]['goals_for'] += ga
                team_stats[opponent_id][next_date]['goals_against'] += gf
                team_stats[opponent_id][next_date]['form'] = (team_stats[opponent_id][next_date]['form'] + [0])[-5:]
                
                if venue == 'Home':
                    # Opponent was away
                    team_stats[opponent_id][next_date]['away_losses'] += 1
                    team_stats[opponent_id][next_date]['away_form'] = (team_stats[opponent_id][next_date]['away_form'] + [0])[-5:]
                else:
                    # Opponent was home
                    team_stats[opponent_id][next_date]['home_losses'] += 1
                    team_stats[opponent_id][next_date]['home_form'] = (team_stats[opponent_id][next_date]['home_form'] + [0])[-5:]
                
                # Update head-to-head - ensure the entry exists first
                if team_id not in team_stats[opponent_id][next_date]['matches_against']:
                    team_stats[opponent_id][next_date]['matches_against'][team_id] = {
                        'played': 0, 'wins': 0, 'draws': 0, 'losses': 0
                    }
                team_stats[opponent_id][next_date]['matches_against'][team_id]['played'] += 1
                team_stats[opponent_id][next_date]['matches_against'][team_id]['losses'] += 1
        
        elif result == 'D':
            # Update team stats for draw
            for next_date in next_match_dates:
                team_stats[team_id][next_date]['played'] += 1
                team_stats[team_id][next_date]['draws'] += 1
                team_stats[team_id][next_date]['points'] += 1
                team_stats[team_id][next_date]['goals_for'] += gf
                team_stats[team_id][next_date]['goals_against'] += ga
                team_stats[team_id][next_date]['form'] = (team_stats[team_id][next_date]['form'] + [1])[-5:]
                
                if venue == 'Home':
                    team_stats[team_id][next_date]['home_draws'] += 1
                    team_stats[team_id][next_date]['home_form'] = (team_stats[team_id][next_date]['home_form'] + [1])[-5:]
                else:
                    team_stats[team_id][next_date]['away_draws'] += 1
                    team_stats[team_id][next_date]['away_form'] = (team_stats[team_id][next_date]['away_form'] + [1])[-5:]
                
                # Update head-to-head - ensure the entry exists first
                if opponent_id not in team_stats[team_id][next_date]['matches_against']:
                    team_stats[team_id][next_date]['matches_against'][opponent_id] = {
                        'played': 0, 'wins': 0, 'draws': 0, 'losses': 0
                    }
                team_stats[team_id][next_date]['matches_against'][opponent_id]['played'] += 1
                team_stats[team_id][next_date]['matches_against'][opponent_id]['draws'] += 1
            
            # Update opponent stats for draw
            for next_date in opponent_next_dates:
                team_stats[opponent_id][next_date]['played'] += 1
                team_stats[opponent_id][next_date]['draws'] += 1
                team_stats[opponent_id][next_date]['points'] += 1
                team_stats[opponent_id][next_date]['goals_for'] += ga
                team_stats[opponent_id][next_date]['goals_against'] += gf
                team_stats[opponent_id][next_date]['form'] = (team_stats[opponent_id][next_date]['form'] + [1])[-5:]
                
                if venue == 'Home':
                    # Opponent was away
                    team_stats[opponent_id][next_date]['away_draws'] += 1
                    team_stats[opponent_id][next_date]['away_form'] = (team_stats[opponent_id][next_date]['away_form'] + [1])[-5:]
                else:
                    # Opponent was home
                    team_stats[opponent_id][next_date]['home_draws'] += 1
                    team_stats[opponent_id][next_date]['home_form'] = (team_stats[opponent_id][next_date]['home_form'] + [1])[-5:]
                
                # Update head-to-head - ensure the entry exists first
                if team_id not in team_stats[opponent_id][next_date]['matches_against']:
                    team_stats[opponent_id][next_date]['matches_against'][team_id] = {
                        'played': 0, 'wins': 0, 'draws': 0, 'losses': 0
                    }
                team_stats[opponent_id][next_date]['matches_against'][team_id]['played'] += 1
                team_stats[opponent_id][next_date]['matches_against'][team_id]['draws'] += 1
        
        elif result == 'L':
            # Update team stats for loss
            for next_date in next_match_dates:
                team_stats[team_id][next_date]['played'] += 1
                team_stats[team_id][next_date]['losses'] += 1
                team_stats[team_id][next_date]['goals_for'] += gf
                team_stats[team_id][next_date]['goals_against'] += ga
                team_stats[team_id][next_date]['form'] = (team_stats[team_id][next_date]['form'] + [0])[-5:]
                
                if venue == 'Home':
                    team_stats[team_id][next_date]['home_losses'] += 1
                    team_stats[team_id][next_date]['home_form'] = (team_stats[team_id][next_date]['home_form'] + [0])[-5:]
                else:
                    team_stats[team_id][next_date]['away_losses'] += 1
                    team_stats[team_id][next_date]['away_form'] = (team_stats[team_id][next_date]['away_form'] + [0])[-5:]
                
                # Update head-to-head - ensure the entry exists first
                if opponent_id not in team_stats[team_id][next_date]['matches_against']:
                    team_stats[team_id][next_date]['matches_against'][opponent_id] = {
                        'played': 0, 'wins': 0, 'draws': 0, 'losses': 0
                    }
                team_stats[team_id][next_date]['matches_against'][opponent_id]['played'] += 1
                team_stats[team_id][next_date]['matches_against'][opponent_id]['losses'] += 1
            
            # Update opponent stats for win
            for next_date in opponent_next_dates:
                team_stats[opponent_id][next_date]['played'] += 1
                team_stats[opponent_id][next_date]['wins'] += 1
                team_stats[opponent_id][next_date]['points'] += 3
                team_stats[opponent_id][next_date]['goals_for'] += ga
                team_stats[opponent_id][next_date]['goals_against'] += gf
                team_stats[opponent_id][next_date]['form'] = (team_stats[opponent_id][next_date]['form'] + [3])[-5:]
                
                if venue == 'Home':
                    # Opponent was away
                    team_stats[opponent_id][next_date]['away_wins'] += 1
                    team_stats[opponent_id][next_date]['away_form'] = (team_stats[opponent_id][next_date]['away_form'] + [3])[-5:]
                else:
                    # Opponent was home
                    team_stats[opponent_id][next_date]['home_wins'] += 1
                    team_stats[opponent_id][next_date]['home_form'] = (team_stats[opponent_id][next_date]['home_form'] + [3])[-5:]
                
                # Update head-to-head - ensure the entry exists first
                if team_id not in team_stats[opponent_id][next_date]['matches_against']:
                    team_stats[opponent_id][next_date]['matches_against'][team_id] = {
                        'played': 0, 'wins': 0, 'draws': 0, 'losses': 0
                    }
                team_stats[opponent_id][next_date]['matches_against'][team_id]['played'] += 1
                team_stats[opponent_id][next_date]['matches_against'][team_id]['wins'] += 1
                
    return team_stats

def get_league_position(team_stats, date):
    """Calculate league positions for all teams at a given date"""
    teams_on_date = {}
    
    # Get stats for all teams on this date or the closest previous date
    for team_id, dates in team_stats.items():
        prev_dates = [d for d in dates.keys() if d <= date]
        if prev_dates:
            closest_date = max(prev_dates)
            teams_on_date[team_id] = {
                'points': team_stats[team_id][closest_date]['points'],
                'goal_diff': team_stats[team_id][closest_date]['goals_for'] - team_stats[team_id][closest_date]['goals_against'],
                'goals_for': team_stats[team_id][closest_date]['goals_for']
            }
    
    # Sort teams by points, goal difference, then goals for
    sorted_teams = sorted(
        teams_on_date.items(),
        key=lambda x: (x[1]['points'], x[1]['goal_diff'], x[1]['goals_for']),
        reverse=True
    )
    
    # Assign positions
    positions = {}
    for pos, (team_id, _) in enumerate(sorted_teams, 1):
        positions[team_id] = pos
    
    return positions

def prepare_match_features(data, team_stats, team_to_id):
    """Create features for each match that would be known before kickoff"""
    features = []
    targets = []
    match_info = []
    
    # Process each match to extract features
    for _, match in data.iterrows():
        date = match['date']
        team = match['team']
        opponent = match['opponent']
        venue = match['venue']
        result = match['result']
        
        team_id = team_to_id[team]
        opponent_id = team_to_id[opponent]
        
        # Find the closest date before the match for team stats
        team_prev_dates = [d for d in team_stats[team_id].keys() if d < date]
        opponent_prev_dates = [d for d in team_stats[opponent_id].keys() if d < date]
        
        # Skip if no previous data is available for either team
        if not team_prev_dates or not opponent_prev_dates:
            continue
        
        team_stats_date = max(team_prev_dates)
        opponent_stats_date = max(opponent_prev_dates)
        
        # Calculate league positions
        league_positions = get_league_position(team_stats, date)
        
        # Create feature vector
        feature = [
            1 if venue == 'Home' else 0,  # Home or away game
            
            # Team stats
            team_stats[team_id][team_stats_date]['points'],
            team_stats[team_id][team_stats_date]['played'],
            team_stats[team_id][team_stats_date]['wins'] / max(1, team_stats[team_id][team_stats_date]['played']),
            team_stats[team_id][team_stats_date]['draws'] / max(1, team_stats[team_id][team_stats_date]['played']),
            team_stats[team_id][team_stats_date]['losses'] / max(1, team_stats[team_id][team_stats_date]['played']),
            team_stats[team_id][team_stats_date]['goals_for'] / max(1, team_stats[team_id][team_stats_date]['played']),
            team_stats[team_id][team_stats_date]['goals_against'] / max(1, team_stats[team_id][team_stats_date]['played']),
            
            # League position
            league_positions.get(team_id, 20) / 20,  # Normalize by total teams
            
            # Form (last 5 games) - average points per game
            sum(team_stats[team_id][team_stats_date]['form']) / max(1, len(team_stats[team_id][team_stats_date]['form'])) / 3,
            
            # Home/Away specific form
            sum(team_stats[team_id][team_stats_date]['home_form'] if venue == 'Home' else 
                team_stats[team_id][team_stats_date]['away_form']) / 
                max(1, len(team_stats[team_id][team_stats_date]['home_form'] if venue == 'Home' 
                      else team_stats[team_id][team_stats_date]['away_form'])) / 3,
            
            # Opponent stats
            team_stats[opponent_id][opponent_stats_date]['points'],
            team_stats[opponent_id][opponent_stats_date]['played'],
            team_stats[opponent_id][opponent_stats_date]['wins'] / max(1, team_stats[opponent_id][opponent_stats_date]['played']),
            team_stats[opponent_id][opponent_stats_date]['draws'] / max(1, team_stats[opponent_id][opponent_stats_date]['played']),
            team_stats[opponent_id][opponent_stats_date]['losses'] / max(1, team_stats[opponent_id][opponent_stats_date]['played']),
            team_stats[opponent_id][opponent_stats_date]['goals_for'] / max(1, team_stats[opponent_id][opponent_stats_date]['played']),
            team_stats[opponent_id][opponent_stats_date]['goals_against'] / max(1, team_stats[opponent_id][opponent_stats_date]['played']),
            
            # Opponent league position
            league_positions.get(opponent_id, 20) / 20,  # Normalize by total teams
            
            # Opponent form
            sum(team_stats[opponent_id][opponent_stats_date]['form']) / max(1, len(team_stats[opponent_id][opponent_stats_date]['form'])) / 3,
            
            # Opponent Home/Away specific form
            sum(team_stats[opponent_id][opponent_stats_date]['away_form'] if venue == 'Home' else 
                team_stats[opponent_id][opponent_stats_date]['home_form']) / 
                max(1, len(team_stats[opponent_id][opponent_stats_date]['away_form'] if venue == 'Home' 
                      else team_stats[opponent_id][opponent_stats_date]['home_form'])) / 3,
            
            # Head-to-head stats
            team_stats[team_id][team_stats_date]['matches_against'].get(opponent_id, {}).get('played', 0),
            team_stats[team_id][team_stats_date]['matches_against'].get(opponent_id, {}).get('wins', 0) / 
                max(1, team_stats[team_id][team_stats_date]['matches_against'].get(opponent_id, {}).get('played', 0)),
            team_stats[team_id][team_stats_date]['matches_against'].get(opponent_id, {}).get('draws', 0) / 
                max(1, team_stats[team_id][team_stats_date]['matches_against'].get(opponent_id, {}).get('played', 0)),
            team_stats[team_id][team_stats_date]['matches_against'].get(opponent_id, {}).get('losses', 0) / 
                max(1, team_stats[team_id][team_stats_date]['matches_against'].get(opponent_id, {}).get('played', 0)),
        ]
        
        features.append(feature)
        
        # Set target based on the result
        if result == 'W':
            target = 0  # Win
        elif result == 'D':
            target = 1  # Draw
        else:  # result == 'L'
            target = 2  # Loss
        
        targets.append(target)
        
        # Store match info for reference
        match_info.append({
            'date': date,
            'team': team,
            'opponent': opponent,
            'venue': venue,
            'result': result
        })
    
    return np.array(features), np.array(targets), match_info

class MatchPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.3):
        super(MatchPredictor, self).__init__()
        
        # More sophisticated architecture with skip connections
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn2 = nn.BatchNorm1d(hidden_size // 2)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.fc3 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.bn3 = nn.BatchNorm1d(hidden_size // 4)
        self.dropout3 = nn.Dropout(dropout_rate)
        
        self.fc4 = nn.Linear(hidden_size // 4, output_size)
        
        # Skip connection components
        self.skip1 = nn.Linear(input_size, hidden_size // 2)
        self.skip2 = nn.Linear(hidden_size, hidden_size // 4)
        
    def forward(self, x):
        # First layer with skip connection to the second layer output
        x1 = self.fc1(x)
        x1 = nn.functional.relu(x1)
        if x1.shape[0] > 1:  # BatchNorm needs more than 1 sample
            x1 = self.bn1(x1)
        x1 = self.dropout1(x1)
        
        # Skip connection from input to second layer
        skip1_out = self.skip1(x)
        
        # Second layer with skip connection
        x2 = self.fc2(x1) + skip1_out
        x2 = nn.functional.relu(x2)
        if x2.shape[0] > 1:
            x2 = self.bn2(x2)
        x2 = self.dropout2(x2)
        
        # Skip connection from first layer to third layer
        skip2_out = self.skip2(x1)  # Use x1 instead of applying fc1 again
        
        # Third layer
        x3 = self.fc3(x2) + skip2_out
        x3 = nn.functional.relu(x3)
        if x3.shape[0] > 1:
            x3 = self.bn3(x3)
        x3 = self.dropout3(x3)
        
        # Output layer
        x4 = self.fc4(x3)
        
        return x4

def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=100, patience=10):
    model.to(device)
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    best_val_accuracy = 0
    epochs_without_improvement = 0
    best_model_state = model.state_dict().copy()
    
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
        
        # Early stopping check
        if epoch_val_accuracy > best_val_accuracy:
            best_val_accuracy = epoch_val_accuracy
            best_model_state = model.state_dict().copy()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        
        if epochs_without_improvement >= patience:
            print(f'Early stopping triggered after {epoch+1} epochs')
            # Restore the best model
            model.load_state_dict(best_model_state)
            break
        
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_train_loss:.4f}, '
                  f'Val Loss: {epoch_val_loss:.4f}, Val Accuracy: {epoch_val_accuracy:.4f}')
    
    # Make sure we use the best model
    model.load_state_dict(best_model_state)
    
    # Plot the training and validation results
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Validation Accuracy')
    
    plt.tight_layout()
    plt.savefig('model/training_results.png')
    
    return model, train_losses, val_losses, val_accuracies

def predict_match(model, team, opponent, is_home, match_date, team_stats, team_to_id, id_to_team, device):
    """
    Prepare prediction data for a future match based on historical stats.
    
    Args:
        model: The trained model
        team: Team name
        opponent: Opponent team name
        is_home: Whether the match is at home (1) or away (0)
        match_date: Date of the match (YYYY-MM-DD)
        team_stats: Historical team statistics
        team_to_id: Mapping from team names to IDs
        id_to_team: Mapping from IDs to team names
        device: Device to run the model on
    """
    try:
        team_id = team_to_id[team]
        opponent_id = team_to_id[opponent]
    except KeyError:
        raise ValueError(f"Team '{team}' or opponent '{opponent}' not found in training data")
    
    # Convert match_date to datetime
    if isinstance(match_date, str):
        match_date = pd.to_datetime(match_date)
    
    # Find the closest date before the match for team stats
    team_prev_dates = [d for d in team_stats[team_id].keys() if d < match_date]
    opponent_prev_dates = [d for d in team_stats[opponent_id].keys() if d < match_date]
    
    if not team_prev_dates or not opponent_prev_dates:
        raise ValueError(f"No historical data available for {team} or {opponent} before {match_date}")
    
    team_stats_date = max(team_prev_dates)
    opponent_stats_date = max(opponent_prev_dates)
    
    # Calculate league positions
    league_positions = get_league_position(team_stats, match_date)
    
    # Create feature vector (same structure as in prepare_match_features)
    feature = [
        1 if is_home else 0,  # Home or away game
        
        # Team stats
        team_stats[team_id][team_stats_date]['points'],
        team_stats[team_id][team_stats_date]['played'],
        team_stats[team_id][team_stats_date]['wins'] / max(1, team_stats[team_id][team_stats_date]['played']),
        team_stats[team_id][team_stats_date]['draws'] / max(1, team_stats[team_id][team_stats_date]['played']),
        team_stats[team_id][team_stats_date]['losses'] / max(1, team_stats[team_id][team_stats_date]['played']),
        team_stats[team_id][team_stats_date]['goals_for'] / max(1, team_stats[team_id][team_stats_date]['played']),
        team_stats[team_id][team_stats_date]['goals_against'] / max(1, team_stats[team_id][team_stats_date]['played']),
        
        # League position
        league_positions.get(team_id, 20) / 20,  # Normalize by total teams
        
        # Form (last 5 games) - average points per game
        sum(team_stats[team_id][team_stats_date]['form']) / max(1, len(team_stats[team_id][team_stats_date]['form'])) / 3,
        
        # Home/Away specific form
        sum(team_stats[team_id][team_stats_date]['home_form'] if is_home else 
            team_stats[team_id][team_stats_date]['away_form']) / 
            max(1, len(team_stats[team_id][team_stats_date]['home_form'] if is_home 
                  else team_stats[team_id][team_stats_date]['away_form'])) / 3,
        
        # Opponent stats
        team_stats[opponent_id][opponent_stats_date]['points'],
        team_stats[opponent_id][opponent_stats_date]['played'],
        team_stats[opponent_id][opponent_stats_date]['wins'] / max(1, team_stats[opponent_id][opponent_stats_date]['played']),
        team_stats[opponent_id][opponent_stats_date]['draws'] / max(1, team_stats[opponent_id][opponent_stats_date]['played']),
        team_stats[opponent_id][opponent_stats_date]['losses'] / max(1, team_stats[opponent_id][opponent_stats_date]['played']),
        team_stats[opponent_id][opponent_stats_date]['goals_for'] / max(1, team_stats[opponent_id][opponent_stats_date]['played']),
        team_stats[opponent_id][opponent_stats_date]['goals_against'] / max(1, team_stats[opponent_id][opponent_stats_date]['played']),
        
        # Opponent league position
        league_positions.get(opponent_id, 20) / 20,  # Normalize by total teams
        
        # Opponent form
        sum(team_stats[opponent_id][opponent_stats_date]['form']) / max(1, len(team_stats[opponent_id][opponent_stats_date]['form'])) / 3,
        
        # Opponent Home/Away specific form
        sum(team_stats[opponent_id][opponent_stats_date]['away_form'] if is_home else 
            team_stats[opponent_id][opponent_stats_date]['home_form']) / 
            max(1, len(team_stats[opponent_id][opponent_stats_date]['away_form'] if is_home 
                  else team_stats[opponent_id][opponent_stats_date]['home_form'])) / 3,
        
        # Head-to-head stats
        team_stats[team_id][team_stats_date]['matches_against'].get(opponent_id, {}).get('played', 0),
        team_stats[team_id][team_stats_date]['matches_against'].get(opponent_id, {}).get('wins', 0) / 
            max(1, team_stats[team_id][team_stats_date]['matches_against'].get(opponent_id, {}).get('played', 0)),
        team_stats[team_id][team_stats_date]['matches_against'].get(opponent_id, {}).get('draws', 0) / 
            max(1, team_stats[team_id][team_stats_date]['matches_against'].get(opponent_id, {}).get('played', 0)),
        team_stats[team_id][team_stats_date]['matches_against'].get(opponent_id, {}).get('losses', 0) / 
            max(1, team_stats[team_id][team_stats_date]['matches_against'].get(opponent_id, {}).get('played', 0)),
    ]
    
    # Convert feature to tensor and make prediction
    features_tensor = torch.tensor([feature], dtype=torch.float32).to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(features_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
        # Get result with highest probability
        predicted_idx = torch.argmax(probabilities, dim=1).item()
        result_mapping = {0: 'Win', 1: 'Draw', 2: 'Loss'}
        predicted_result = result_mapping[predicted_idx]
        
        # Return predicted result and probabilities
        return predicted_result, {
            'Win': probabilities[0][0].item(),
            'Draw': probabilities[0][1].item(),
            'Loss': probabilities[0][2].item()
        }

def evaluate_model(model, test_loader, device):
    """Evaluate model performance on the test set with detailed metrics"""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # Calculate accuracy
    accuracy = (all_preds == all_targets).mean()
    
    # Calculate class-specific metrics
    class_names = ['Win', 'Draw', 'Loss']
    class_metrics = {}
    
    for i, class_name in enumerate(class_names):
        # True positives, false positives, false negatives
        tp = ((all_preds == i) & (all_targets == i)).sum()
        fp = ((all_preds == i) & (all_targets != i)).sum()
        fn = ((all_preds != i) & (all_targets == i)).sum()
        
        # Precision, recall, F1
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-10)
        
        class_metrics[class_name] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': (all_targets == i).sum()
        }
    
    # Create confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(all_targets, all_preds)
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Add text annotations
    fmt = '.2f'
    thresh = cm_normalized.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, f"{cm[i, j]}\n({cm_normalized[i, j]:.2f})",
                     ha="center", va="center",
                     color="white" if cm_normalized[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('model/confusion_matrix.png')
    
    return {
        'accuracy': accuracy,
        'class_metrics': class_metrics,
        'confusion_matrix': cm,
        'confusion_matrix_normalized': cm_normalized
    }

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    print("Loading Premier League data...")
    try:
        data = pd.read_csv('model/dataset/FootballMatches.csv')
    except FileNotFoundError:
        csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
        if csv_files:
            print(f"Premier League data not found, using {csv_files[0]} instead")
            data = pd.read_csv(csv_files[0])
        else:
            raise FileNotFoundError("No CSV files found in the current directory")
    
    # Ensure required columns exist
    required_columns = ['date', 'team', 'opponent', 'venue', 'result']
    for col in required_columns:
        if col not in data.columns:
            raise ValueError(f"Required column '{col}' not found in the dataset")
    
    print("Data loaded successfully with shape:", data.shape)
    
    # Create team mapping
    team_to_id, id_to_team = create_team_mapping(data)
    print(f"Found {len(team_to_id)} teams in the dataset")
    
    # Calculate historical team statistics
    print("Calculating team statistics...")
    team_stats = calculate_team_stats(data, team_to_id)
    
    # Prepare features for matches
    print("Preparing match features...")
    features, targets, match_info = prepare_match_features(data, team_stats, team_to_id)
    print(f"Prepared features for {len(features)} matches")
    
    # Split the data (leaving some out for final testing)
    X_temp, X_test, y_temp, y_test, info_temp, info_test = train_test_split(
        features, targets, match_info, test_size=0.15, random_state=42
    )
    
    # Split the remaining data into training and validation sets
    X_train, X_val, y_train, y_val, info_train, info_val = train_test_split(
        X_temp, y_temp, info_temp, test_size=0.2, random_state=42
    )
    
    # Scale the features for better model performance
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # Create datasets and dataloaders
    train_dataset = FootballDataset(X_train, y_train)
    val_dataset = FootballDataset(X_val, y_val)
    test_dataset = FootballDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Initialize the model
    input_size = X_train.shape[1]
    hidden_size = 128
    output_size = 3  # Win, Draw, Loss
    
    model = MatchPredictor(input_size, hidden_size, output_size)
    print(model)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # Train the model
    print("Starting training...")
    model, train_losses, val_losses, val_accuracies = train_model(
        model, train_loader, val_loader, criterion, optimizer, device, 
        num_epochs=100, patience=15
    )
    
    # Evaluate the model on the test set
    print("\nEvaluating on test set...")
    evaluation_metrics = evaluate_model(model, test_loader, device)
    
    print(f"Test Accuracy: {evaluation_metrics['accuracy']:.4f}")
    print("\nClass-specific metrics:")
    for class_name, metrics in evaluation_metrics['class_metrics'].items():
        print(f"{class_name}: Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, F1={metrics['f1']:.4f}")
    
    # Create directories if they don't exist
    os.makedirs('model/saved_model', exist_ok=True)
    
    # Save the model
    torch.save(model.state_dict(), 'model/saved_model/premier_league_predictor.pth')
    
    # Save the preprocessor, encoders, and other components
    with open('model/saved_model/model_components.pkl', 'wb') as f:
        pickle.dump({
            'team_to_id': team_to_id,
            'id_to_team': id_to_team,
            'team_stats': team_stats,
            'scaler': scaler,
            'input_size': input_size,
            'hidden_size': hidden_size,
            'output_size': output_size
        }, f)
    
    print("\nModel and components saved in 'model/saved_model' directory")
    
    # Example prediction
    print("\nExample prediction:")
    # Use the first test match as an example
    example_match = info_test[0]
    
    # Make prediction using historical data
    prediction, probabilities = predict_match(
        model,
        example_match['team'],
        example_match['opponent'],
        example_match['venue'] == 'Home',
        example_match['date'],
        team_stats,
        team_to_id,
        id_to_team,
        device
    )
    
    print(f"Match: {example_match['team']} vs {example_match['opponent']} ({example_match['venue']})")
    print(f"Actual result: {example_match['result']}")
    print(f"Predicted result: {prediction}")
    print(f"Prediction probabilities: Win={probabilities['Win']:.2f}, Draw={probabilities['Draw']:.2f}, Loss={probabilities['Loss']:.2f}")

if __name__ == "__main__":
    main()
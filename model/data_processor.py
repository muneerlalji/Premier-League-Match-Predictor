import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader

class PremierLeagueDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    
    result_mapping = {'H': 0, 'D': 1, 'A': 2}
    df['FTR_encoded'] = df['FTR'].map(result_mapping)
    
    df = df.dropna(subset=['HomeGoals', 'AwayGoals', 'FTR', 'FTR_encoded'])
    
    teams = df['Home'].unique()
    
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    
    team_stats = {}
    for team in teams:
        team_stats[team] = {
            'last_5_results': [],
            'goals_scored_last_5': [],
            'goals_conceded_last_5': [],
        }
    
    processed_data = []
    for _, match in df.iterrows():
        home_team = match['Home']
        away_team = match['Away']
        
        has_home_data = home_team in team_stats and len(team_stats[home_team]['last_5_results']) >= 3
        has_away_data = away_team in team_stats and len(team_stats[away_team]['last_5_results']) >= 3
        
        if has_home_data and has_away_data:
            home_goal_diff = np.mean(team_stats[home_team]['goals_scored_last_5']) - np.mean(team_stats[home_team]['goals_conceded_last_5'])
            away_goal_diff = np.mean(team_stats[away_team]['goals_scored_last_5']) - np.mean(team_stats[away_team]['goals_conceded_last_5'])
            
            feature = [
                np.mean(team_stats[home_team]['last_5_results']),
                np.mean(team_stats[home_team]['goals_scored_last_5']),
                np.mean(team_stats[home_team]['goals_conceded_last_5']),
                
                np.mean(team_stats[away_team]['last_5_results']),
                np.mean(team_stats[away_team]['goals_scored_last_5']),
                np.mean(team_stats[away_team]['goals_conceded_last_5']),
                
                home_goal_diff,
                away_goal_diff,
                home_goal_diff - away_goal_diff,
            ]
            
            processed_data.append({
                'feature': feature,
                'label': match['FTR_encoded'],
                'match_info': f"{match['Home']} vs {match['Away']} ({match['Date']})"
            })
        
        home_result = 2 if match['FTR'] == 'H' else (1 if match['FTR'] == 'D' else 0)
        away_result = 2 if match['FTR'] == 'A' else (1 if match['FTR'] == 'D' else 0)
        
        if home_team not in team_stats:
            team_stats[home_team] = {
                'last_5_results': [],
                'goals_scored_last_5': [],
                'goals_conceded_last_5': [],
            }
        
        team_stats[home_team]['last_5_results'].append(home_result)
        team_stats[home_team]['goals_scored_last_5'].append(match['HomeGoals'])
        team_stats[home_team]['goals_conceded_last_5'].append(match['AwayGoals'])
        
        for key in ['last_5_results', 'goals_scored_last_5', 'goals_conceded_last_5']:
            team_stats[home_team][key] = team_stats[home_team][key][-5:]
        
        if away_team not in team_stats:
            team_stats[away_team] = {
                'last_5_results': [],
                'goals_scored_last_5': [],
                'goals_conceded_last_5': [],
            }
        
        team_stats[away_team]['last_5_results'].append(away_result)
        team_stats[away_team]['goals_scored_last_5'].append(match['AwayGoals'])
        team_stats[away_team]['goals_conceded_last_5'].append(match['HomeGoals'])
        
        for key in ['last_5_results', 'goals_scored_last_5', 'goals_conceded_last_5']:
            team_stats[away_team][key] = team_stats[away_team][key][-5:]
    
    X = np.array([data['feature'] for data in processed_data])
    y = np.array([data['label'] for data in processed_data])
    
    unique_values, counts = np.unique(y, return_counts=True)
    print("Overall dataset class distribution:")
    for val, count in zip(unique_values, counts):
        result_name = ['Home Win', 'Draw', 'Away Win'][val]
        print(f"{result_name}: {count} ({count/len(y)*100:.2f}%)")
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    train_dataset = PremierLeagueDataset(X_train, y_train)
    test_dataset = PremierLeagueDataset(X_test, y_test)
    
    return train_dataset, test_dataset, scaler

def create_data_loaders(train_dataset, test_dataset, batch_size=64):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

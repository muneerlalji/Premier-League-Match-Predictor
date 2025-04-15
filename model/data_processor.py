import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict

class PremierLeagueDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def calculate_league_positions(df):
    df = df.copy()
    seasons = df['Season_End_Year'].unique()
    
    team_positions = {}
    
    for season in seasons:
        season_df = df[df['Season_End_Year'] == season].copy()
        
        season_df['Date'] = pd.to_datetime(season_df['Date'])
        season_df = season_df.sort_values('Date')
        
        teams = set(season_df['Home'].unique()) | set(season_df['Away'].unique())
        standings = {team: {'points': 0, 'goal_diff': 0, 'played': 0} for team in teams}
        
        positions_by_date = {}
        
        for _, match in season_df.iterrows():
            date = match['Date']
            home_team = match['Home']
            away_team = match['Away']
            
            if 'FTR' in match and pd.notna(match['FTR']):
                home_goals = match['HomeGoals'] if 'HomeGoals' in match else 0
                away_goals = match['AwayGoals'] if 'AwayGoals' in match else 0
                
                standings[home_team]['goal_diff'] += (home_goals - away_goals)
                standings[away_team]['goal_diff'] += (away_goals - home_goals)
                
                if match['FTR'] == 'H':  # Home win
                    standings[home_team]['points'] += 3
                elif match['FTR'] == 'A':  # Away win
                    standings[away_team]['points'] += 3
                else:  # Draw
                    standings[home_team]['points'] += 1
                    standings[away_team]['points'] += 1
                
                standings[home_team]['played'] += 1
                standings[away_team]['played'] += 1
                
            current_standings = sorted(
                standings.items(),
                key=lambda x: (x[1]['points'], x[1]['goal_diff']),
                reverse=True
            )
            
            positions = {team: pos+1 for pos, (team, _) in enumerate(current_standings)}
            
            date_str = date.strftime('%Y-%m-%d')
            positions_by_date[date_str] = positions
        
        team_positions[season] = positions_by_date
    
    return team_positions

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    
    result_mapping = {'H': 0, 'D': 1, 'A': 2}
    df['FTR_encoded'] = df['FTR'].map(result_mapping)
    
    df = df.dropna(subset=['HomeGoals', 'AwayGoals', 'FTR', 'FTR_encoded'])
    
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    
    league_positions = calculate_league_positions(df)
    
    teams = df['Home'].unique()
    
    team_stats = {}
    for team in teams:
        team_stats[team] = {
            'last_5_results': [],
            'goals_scored_last_5': [],
            'goals_conceded_last_5': [],
            'home_results_last_5': [],
            'away_results_last_5': [],
            'home_goals_scored_last_5': [],
            'home_goals_conceded_last_5': [],
            'away_goals_scored_last_5': [],
            'away_goals_conceded_last_5': [],
        }
    
    processed_data = []
    for idx, match in df.iterrows():
        home_team = match['Home']
        away_team = match['Away']
        
        match_date = match['Date'].strftime('%Y-%m-%d')
        season = match['Season_End_Year']

        home_position = league_positions[season][match_date].get(home_team, None)
        away_position = league_positions[season][match_date].get(away_team, None)

        has_home_data = home_team in team_stats and len(team_stats[home_team]['last_5_results']) >= 3
        has_away_data = away_team in team_stats and len(team_stats[away_team]['last_5_results']) >= 3
        
        has_home_specific = has_home_data and len(team_stats[home_team]['home_results_last_5']) >= 2
        has_away_specific = has_away_data and len(team_stats[away_team]['away_results_last_5']) >= 2
        
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
            
            if has_home_specific and has_away_specific:
                home_home_form = np.mean(team_stats[home_team]['home_results_last_5'])
                home_home_goals_scored = np.mean(team_stats[home_team]['home_goals_scored_last_5'])
                home_home_goals_conceded = np.mean(team_stats[home_team]['home_goals_conceded_last_5'])
                
                away_away_form = np.mean(team_stats[away_team]['away_results_last_5'])
                away_away_goals_scored = np.mean(team_stats[away_team]['away_goals_scored_last_5'])
                away_away_goals_conceded = np.mean(team_stats[away_team]['away_goals_conceded_last_5'])
                
                feature.extend([
                    home_home_form,
                    home_home_goals_scored,
                    home_home_goals_conceded,
                    away_away_form,
                    away_away_goals_scored,
                    away_away_goals_conceded,
                ])
            else:
                # Add placeholder values if specific form not available
                feature.extend([0, 0, 0, 0, 0, 0])
            
            feature.append(home_position)
            feature.append(away_position)
            feature.append(home_position - away_position)
            
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
                'home_results_last_5': [],
                'away_results_last_5': [],
                'home_goals_scored_last_5': [],
                'home_goals_conceded_last_5': [],
                'away_goals_scored_last_5': [],
                'away_goals_conceded_last_5': [],
            }
        
        if away_team not in team_stats:
            team_stats[away_team] = {
                'last_5_results': [],
                'goals_scored_last_5': [],
                'goals_conceded_last_5': [],
                'home_results_last_5': [],
                'away_results_last_5': [],
                'home_goals_scored_last_5': [],
                'home_goals_conceded_last_5': [],
                'away_goals_scored_last_5': [],
                'away_goals_conceded_last_5': [],
            }
        
        team_stats[home_team]['last_5_results'].append(home_result)
        team_stats[home_team]['goals_scored_last_5'].append(match['HomeGoals'])
        team_stats[home_team]['goals_conceded_last_5'].append(match['AwayGoals'])
        
        team_stats[away_team]['last_5_results'].append(away_result)
        team_stats[away_team]['goals_scored_last_5'].append(match['AwayGoals'])
        team_stats[away_team]['goals_conceded_last_5'].append(match['HomeGoals'])
        
        team_stats[home_team]['home_results_last_5'].append(home_result)
        team_stats[home_team]['home_goals_scored_last_5'].append(match['HomeGoals'])
        team_stats[home_team]['home_goals_conceded_last_5'].append(match['AwayGoals'])
        
        team_stats[away_team]['away_results_last_5'].append(away_result)
        team_stats[away_team]['away_goals_scored_last_5'].append(match['AwayGoals'])
        team_stats[away_team]['away_goals_conceded_last_5'].append(match['HomeGoals'])
        
        for key in team_stats[home_team].keys():
            team_stats[home_team][key] = team_stats[home_team][key][-5:]
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
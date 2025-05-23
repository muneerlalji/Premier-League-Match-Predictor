import torch
import pandas as pd
import numpy as np
import os
import pickle
from data_processor import PremierLeagueDataset
from train_model import MatchPredictor
from sklearn.preprocessing import StandardScaler
from torch.serialization import add_safe_globals

def load_model(model_path, device):
    loaded_model = torch.load(model_path, map_location=device, weights_only=False)
    
    input_size = 18
    hidden_size = 128
    num_classes = 3
    model = MatchPredictor(input_size, hidden_size, num_classes).to(device)
    
    # Load model weights
    model.load_state_dict(loaded_model['model_state_dict'])
    
    model.eval()
    
    scaler = loaded_model['scaler']
    
    return model, scaler

def predict_match(model, scaler, home_team, away_team, team_stats, device):
    if home_team not in team_stats or away_team not in team_stats:
        return "Insufficient data for one or both teams"
    
    home_stats = team_stats[home_team]
    away_stats = team_stats[away_team]
    
    features = [
        np.mean(home_stats['last_5_results']) if len(home_stats['last_5_results']) > 0 else 0,
        np.mean(home_stats['goals_scored_last_5']) if len(home_stats['goals_scored_last_5']) > 0 else 0,
        np.mean(home_stats['goals_conceded_last_5']) if len(home_stats['goals_conceded_last_5']) > 0 else 0,
        np.std(home_stats['last_5_results']) if len(home_stats['last_5_results']) > 1 else 0,
        np.std(home_stats['goals_scored_last_5']) if len(home_stats['goals_scored_last_5']) > 1 else 0,
        np.std(home_stats['goals_conceded_last_5']) if len(home_stats['goals_conceded_last_5']) > 1 else 0, 
        np.max(home_stats['last_5_results']) if len(home_stats['last_5_results']) > 0 else 0,
        np.min(home_stats['last_5_results']) if len(home_stats['last_5_results']) > 0 else 0,
        len(home_stats['last_5_results']),
        
        np.mean(away_stats['last_5_results']) if len(away_stats['last_5_results']) > 0 else 0,
        np.mean(away_stats['goals_scored_last_5']) if len(away_stats['goals_scored_last_5']) > 0 else 0,
        np.mean(away_stats['goals_conceded_last_5']) if len(away_stats['goals_conceded_last_5']) > 0 else 0,
        np.std(away_stats['last_5_results']) if len(away_stats['last_5_results']) > 1 else 0,
        np.std(away_stats['goals_scored_last_5']) if len(away_stats['goals_scored_last_5']) > 1 else 0,
        np.std(away_stats['goals_conceded_last_5']) if len(away_stats['goals_conceded_last_5']) > 1 else 0,
        np.max(away_stats['last_5_results']) if len(away_stats['last_5_results']) > 0 else 0,
        np.min(away_stats['last_5_results']) if len(away_stats['last_5_results']) > 0 else 0,
        len(away_stats['last_5_results'])
    ]
    
    scaled_features = scaler.transform(np.array([features]))
    
    input_tensor = torch.tensor(scaled_features, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)[0]
        prediction = torch.argmax(output, dim=1).item()
    
    result_mapping = {0: 'Home Win', 1: 'Draw', 2: 'Away Win'}
    predicted_result = result_mapping[prediction]
    
    prob_dict = {
        'Home Win': probabilities[0].item() * 100,
        'Draw': probabilities[1].item() * 100,
        'Away Win': probabilities[2].item() * 100
    }
    
    return {
        'prediction': predicted_result,
        'probabilities': prob_dict,
        'match': f"{home_team} vs {away_team}"
    }

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model_path = 'model/saved_model/premier_league_model.pth'
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found. Please train the model first.")
        return
    
    model, scaler = load_model(model_path, device)
    
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                           'model/dataset/Premier League-Matches-1993-2023.csv')
    df = pd.read_csv(data_path)
    
    team_stats = {}
    for _, match in df.iterrows():
        home_team = match['Home']
        away_team = match['Away']
        
        result_mapping = {'H': 0, 'D': 1, 'A': 2}
        
        ftr = match.get('FTR', None)
        if ftr is None:
            continue
            
        home_result = result_mapping.get(ftr, None)
        if home_result is None:
            continue
            
        away_result = 2 - home_result if home_result != 1 else 1
        
        if home_team not in team_stats:
            team_stats[home_team] = {
                'last_5_results': [],
                'goals_scored_last_5': [],
                'goals_conceded_last_5': []
            }
        
        home_goals = match.get('HomeGoals', 0)
        away_goals = match.get('AwayGoals', 0)
        
        team_stats[home_team]['last_5_results'].append(home_result)
        team_stats[home_team]['goals_scored_last_5'].append(home_goals)
        team_stats[home_team]['goals_conceded_last_5'].append(away_goals)
        
        team_stats[home_team]['last_5_results'] = team_stats[home_team]['last_5_results'][-5:]
        team_stats[home_team]['goals_scored_last_5'] = team_stats[home_team]['goals_scored_last_5'][-5:]
        team_stats[home_team]['goals_conceded_last_5'] = team_stats[home_team]['goals_conceded_last_5'][-5:]
        
        if away_team not in team_stats:
            team_stats[away_team] = {
                'last_5_results': [],
                'goals_scored_last_5': [],
                'goals_conceded_last_5': []
            }
        
        team_stats[away_team]['last_5_results'].append(away_result)
        team_stats[away_team]['goals_scored_last_5'].append(away_goals)
        team_stats[away_team]['goals_conceded_last_5'].append(home_goals)
        
        team_stats[away_team]['last_5_results'] = team_stats[away_team]['last_5_results'][-5:]
        team_stats[away_team]['goals_scored_last_5'] = team_stats[away_team]['goals_scored_last_5'][-5:]
        team_stats[away_team]['goals_conceded_last_5'] = team_stats[away_team]['goals_conceded_last_5'][-5:]
    
    teams = sorted(list(team_stats.keys()))
    print(f"Available teams: {len(teams)} teams found")
    for i, team in enumerate(teams):
        print(f"{i+1}. {team}")
    
    while True:
        print("\nEnter the index of the home team (0 to quit):")
        try:
            home_idx_input = input().strip()
            if not home_idx_input:
                print("No input provided, please enter a number")
                continue
                
            home_idx = int(home_idx_input) - 1
            if home_idx == -1:
                break
                
            if home_idx < 0 or home_idx >= len(teams):
                print(f"Invalid team index. Please enter a number between 1 and {len(teams)}")
                continue
                
            home_team = teams[home_idx]
            print(f"Selected home team: {home_team}")
            
            print("Enter the index of the away team:")
            away_idx_input = input().strip()
            if not away_idx_input:
                print("No input provided, please enter a number")
                continue
                
            away_idx = int(away_idx_input) - 1
            if away_idx < 0 or away_idx >= len(teams):
                print(f"Invalid team index. Please enter a number between 1 and {len(teams)}")
                continue
                
            away_team = teams[away_idx]
            print(f"Selected away team: {away_team}")
            
            if home_team == away_team:
                print("Home and away teams cannot be the same")
                continue
                
            result = predict_match(model, scaler, home_team, away_team, team_stats, device)
            print(f"\nMatch: {result['match']}")
            print(f"Prediction: {result['prediction']}")
            print(f"Probability breakdown:")
            for outcome, prob in result['probabilities'].items():
                print(f"  {outcome}: {prob:.2f}%")
        except ValueError as ve:
            print(f"Invalid input: {ve}. Please enter a valid number.")
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()

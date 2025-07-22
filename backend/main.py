from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import os
import sys
import torch
import requests
from datetime import datetime, timedelta
import json

# Add the model directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'model'))

from make_prediction import load_model, predict_match

app = FastAPI(title="Premier League Match Predictor API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Load the model and scaler
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'saved_model', 'premier_league_model.pth')
model, scaler = load_model(model_path, device)

# Football-data.org API configuration
FOOTBALL_DATA_API_KEY = os.getenv('FOOTBALL_DATA_API_KEY')
if not FOOTBALL_DATA_API_KEY:
    raise ValueError("FOOTBALL_DATA_API_KEY environment variable is not set")

TEAM_NAME_MAP = {
    "Arsenal FC": "Arsenal",
    "Aston Villa FC": "Aston Villa",
    "AFC Bournemouth": "Bournemouth",
    "Brentford FC": "Brentford",
    "Brighton & Hove Albion FC": "Brighton",
    "Burnley FC": "Burnley",
    "Chelsea FC": "Chelsea",
    "Crystal Palace FC": "Crystal Palace",
    "Everton FC": "Everton",
    "Fulham FC": "Fulham",
    "Ipswich Town FC": "Ipswich Town",
    "Leeds United FC": "Leeds United",
    "Leicester City FC": "Leicester City",
    "Liverpool FC": "Liverpool",
    "Luton Town FC": "Luton Town",
    "Manchester City FC": "Manchester City",
    "Manchester United FC": "Manchester Utd",
    "Newcastle United FC": "Newcastle Utd",
    "Nottingham Forest FC": "Nott'ham Forest",
    "Sheffield United FC": "Sheffield Utd",
    "Southampton FC": "Southampton",
    "Tottenham Hotspur FC": "Tottenham",
    "West Ham United FC": "West Ham",
    "Wolverhampton Wanderers FC": "Wolves",
}

class MatchPrediction(BaseModel):
    home_team: str
    away_team: str
    prediction: str
    probabilities: Dict[str, float]
    match: str

class UpcomingMatch(BaseModel):
    id: int
    home_team: str
    away_team: str
    date: str
    status: str
    competition: str

@app.get("/")
async def root():
    return {"message": "Premier League Match Predictor API"}

@app.get("/upcoming-matches", response_model=List[UpcomingMatch])
async def get_upcoming_matches():
    """Get upcoming Premier League matches from football-data.org API"""
    headers = {'X-Auth-Token': FOOTBALL_DATA_API_KEY}
    
    date_from = '2024-08-14'
    date_to = '2024-08-21'
    
    url = f"http://api.football-data.org/v4/matches?dateFrom={date_from}&dateTo={date_to}&competitions=PL"
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        
        matches = []
        for match in data.get('matches', []):
            matches.append(UpcomingMatch(
                id=match['id'],
                home_team=match['homeTeam']['name'],
                away_team=match['awayTeam']['name'],
                date=match['utcDate'],
                status=match['status'],
                competition=match['competition']['name']
            ))
        return matches
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error fetching matches: {str(e)}")

@app.post("/predict", response_model=MatchPrediction)
async def predict(match: UpcomingMatch):
    """Make a prediction for a given match"""
    try:
        data_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'dataset', 'Premier League-Matches-1993-2023.csv')
        import pandas as pd
        df = pd.read_csv(data_path)
        
        team_stats = {}
        for _, match_data in df.iterrows():
            home_team = match_data['Home']
            away_team = match_data['Away']
            
            result_mapping = {'H': 0, 'D': 1, 'A': 2}
            
            ftr = match_data.get('FTR', None)
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
            
            home_goals = match_data.get('HomeGoals', 0)
            away_goals = match_data.get('AwayGoals', 0)
            
            team_stats[home_team]['last_5_results'].append(home_result)
            team_stats[home_team]['goals_scored_last_5'].append(home_goals)
            team_stats[home_team]['goals_conceded_last_5'].append(away_goals)
            
            team_stats[home_team]['last_5_results'] = team_stats[home_team]['last_5_results'][-5:]
            team_stats[home_team]['goals_scored_last_5'] = team_stats[home_team]['goals_scored_last_5'][-5:]
            team_stats[home_team]['goals_conceded_last_5'] = team_stats[home_team]['goals_conceded_last_5'][-5:]
            
            # Process away team stats
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
        
        # Make prediction
        print(TEAM_NAME_MAP.get(match.home_team))
        print(TEAM_NAME_MAP.get(match.away_team))
        result = predict_match(model, scaler, TEAM_NAME_MAP.get(match.home_team), TEAM_NAME_MAP.get(match.away_team), team_stats, device)
        
        if isinstance(result, str):
            raise HTTPException(status_code=400, detail=result)
            
        return MatchPrediction(
            home_team=match.home_team,
            away_team=match.away_team,
            prediction=result['prediction'],
            probabilities=result['probabilities'],
            match=result['match']
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 
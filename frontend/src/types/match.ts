export interface UpcomingMatch {
    id: number;
    home_team: string;
    away_team: string;
    date: string;
    status: string;
    competition: string;
}

export interface MatchPrediction {
    home_team: string;
    away_team: string;
    prediction: string;
    probabilities: {
        'Home Win': number;
        'Draw': number;
        'Away Win': number;
    };
    match: string;
} 
import axios from 'axios';
import { UpcomingMatch, MatchPrediction } from '../types/match';

const API_BASE_URL = 'http://localhost:8000';

const api = axios.create({
    baseURL: API_BASE_URL,
    headers: {
        'Content-Type': 'application/json',
    },
});

export const getUpcomingMatches = async (): Promise<UpcomingMatch[]> => {
    const response = await api.get<UpcomingMatch[]>('/upcoming-matches');
    return response.data;
};

export const getPrediction = async (match: UpcomingMatch): Promise<MatchPrediction> => {
    const response = await api.post<MatchPrediction>('/predict', match);
    return response.data;
}; 
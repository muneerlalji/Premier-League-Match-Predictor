import React, { useState } from 'react';
import {
    Card,
    CardContent,
    Typography,
    Button,
    CircularProgress,
    Box,
    Stack,
    Paper,
} from '@mui/material';
import { format } from 'date-fns';
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip } from 'recharts';
import { UpcomingMatch, MatchPrediction } from '../types/match';
import { getPrediction } from '../services/api';

interface MatchCardProps {
    match: UpcomingMatch;
}

const COLORS = ['#4caf50', '#ff9800', '#f44336'];

const MatchCard: React.FC<MatchCardProps> = ({ match }) => {
    const [prediction, setPrediction] = useState<MatchPrediction | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const handlePredict = async () => {
        setLoading(true);
        setError(null);
        try {
            const result = await getPrediction(match);
            setPrediction(result);
        } catch (err) {
            setError('Failed to get prediction. Please try again.');
            console.error('Prediction error:', err);
        } finally {
            setLoading(false);
        }
    };

    const formatDate = (dateString: string) => {
        return format(new Date(dateString), 'MMM d, yyyy HH:mm');
    };

    const renderPredictionChart = () => {
        if (!prediction) return null;

        const data = Object.entries(prediction.probabilities).map(([name, value]) => ({
            name,
            value,
        }));

        return (
            <Box sx={{ height: 200, width: '100%', mt: 2 }}>
                <ResponsiveContainer>
                    <PieChart>
                        <Pie
                            data={data}
                            cx="50%"
                            cy="50%"
                            labelLine={false}
                            label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                            outerRadius={80}
                            fill="#8884d8"
                            dataKey="value"
                        >
                            {data.map((entry, index) => (
                                <Cell key={`cell-${index}`} fill={COLORS[index]} />
                            ))}
                        </Pie>
                        <Tooltip />
                    </PieChart>
                </ResponsiveContainer>
            </Box>
        );
    };

    return (
        <Card sx={{ mb: 2, maxWidth: 600, mx: 'auto' }}>
            <CardContent>
                <Stack spacing={2}>
                    <Box>
                        <Typography variant="h6" component="div" align="center">
                            {match.home_team} vs {match.away_team}
                        </Typography>
                        <Typography variant="body2" color="text.secondary" align="center">
                            {formatDate(match.date)}
                        </Typography>
                        <Typography variant="body2" color="text.secondary" align="center">
                            Status: {match.status}
                        </Typography>
                    </Box>

                    {!prediction && !loading && (
                        <Box sx={{ textAlign: 'center' }}>
                            <Button
                                variant="contained"
                                color="primary"
                                onClick={handlePredict}
                                disabled={loading}
                            >
                                Get Prediction
                            </Button>
                        </Box>
                    )}

                    {loading && (
                        <Box sx={{ textAlign: 'center' }}>
                            <CircularProgress />
                        </Box>
                    )}

                    {error && (
                        <Box>
                            <Typography color="error" align="center">
                                {error}
                            </Typography>
                        </Box>
                    )}

                    {prediction && (
                        <Box>
                            <Paper elevation={2} sx={{ p: 2, mt: 2 }}>
                                <Typography variant="h6" align="center" gutterBottom>
                                    Prediction: {prediction.prediction}
                                </Typography>
                                {renderPredictionChart()}
                            </Paper>
                        </Box>
                    )}
                </Stack>
            </CardContent>
        </Card>
    );
};

export default MatchCard; 
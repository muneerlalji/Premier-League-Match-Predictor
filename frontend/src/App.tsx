import React, { useEffect, useState } from 'react';
import {
    Container,
    Typography,
    Box,
    CircularProgress,
    Alert,
    AppBar,
    Toolbar,
} from '@mui/material';
import { getUpcomingMatches } from './services/api';
import { UpcomingMatch } from './types/match';
import MatchCard from './components/MatchCard';

function App() {
    const [matches, setMatches] = useState<UpcomingMatch[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        const fetchMatches = async () => {
            try {
                const data = await getUpcomingMatches();
                setMatches(data);
            } catch (err) {
                setError('Failed to fetch upcoming matches. Please try again later.');
                console.error('Error fetching matches:', err);
            } finally {
                setLoading(false);
            }
        };

        fetchMatches();
    }, []);

    return (
        <Box sx={{ flexGrow: 1 }}>
            <AppBar position="static">
                <Toolbar>
                    <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
                        Premier League Match Predictor
                    </Typography>
                </Toolbar>
            </AppBar>

            <Container maxWidth="md" sx={{ mt: 4, mb: 4 }}>
                {loading && (
                    <Box sx={{ display: 'flex', justifyContent: 'center', mt: 4 }}>
                        <CircularProgress />
                    </Box>
                )}

                {error && (
                    <Alert severity="error" sx={{ mt: 2 }}>
                        {error}
                    </Alert>
                )}

                {!loading && !error && matches.length === 0 && (
                    <Alert severity="info" sx={{ mt: 2 }}>
                        No upcoming matches found.
                    </Alert>
                )}

                {matches.map((match) => (
                    <MatchCard key={match.id} match={match} />
                ))}
            </Container>
        </Box>
    );
}

export default App;

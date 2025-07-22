# Premier League Match Predictor

A machine learning model that predicts the outcome of Premier League matches using historical data.

## Overview

This project uses a neural network model to predict the outcome of Premier League football matches (Home Win, Draw, or Away Win) based on team performance statistics. The model is trained on Premier League match data from 1993-2023.

## Features

- Neural network prediction model with multiple hidden layers
- Batch normalization and dropout for improved performance
- Early stopping to prevent overfitting
- Class weighting to handle imbalanced data
- Evaluation metrics including precision, recall, and F1-score
- Confusion matrix visualization

## Project Structure

```
Premier-League-Match-Predictor/
├── model/
│   ├── dataset/
│   │   └── Premier League-Matches-1993-2023.csv
│   ├── saved_model/
│   │   └── premier_league_model.pth
│   ├── data_processor.py
│   ├── make_prediction.py
│   └── train_model.py
├── README.md
```

## Requirements

- Python >=3.9
- PyTorch
- NumPy
- Pandas
- scikit-learn
- Matplotlib

## Setup and Installation

1. Clone this repository
2. Install the required dependencies:
   ```
   pip install torch numpy pandas scikit-learn matplotlib
   ```
3. Ensure the dataset is in the correct location or update the path in the code

### Running the Application

The application consists of a frontend React application and a FastAPI backend. You'll need to run both services:

#### Backend Setup
1. Navigate to the backend directory:
   ```
   cd backend
   ```
2. Create a virtual environment (recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```
3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Start the backend server:
   ```
   uvicorn main:app --reload
   ```
   The backend will run on http://localhost:8000

#### Frontend Setup
1. Navigate to the frontend directory:
   ```
   cd frontend
   ```
2. Install the required dependencies:
   ```
   npm install
   ```
3. Start the development server:
   ```
   npm start
   ```
   The frontend will run on http://localhost:3000

### Training the Model

Run the training script to train and save the model:

```
python model/train_model.py
```

This will:
- Load and preprocess the Premier League match data
- Train the neural network model
- Generate a training loss graph
- Create a confusion matrix for model evaluation
- Save the trained model to `model/saved_model/premier_league_model.pth`

### Making Predictions

To predict match outcomes:

```
python model/make_prediction.py
```

This interactive script will:
1. Load the trained model
2. Display a list of all teams in the dataset
3. Prompt you to select home and away teams
4. Predict the match outcome and display probability scores for each possible result

## Model Architecture

The neural network consists of:
- Input layer with 18 features
- Three hidden layers with ReLU activation
- Batch normalization after each hidden layer
- Dropout (30%) for regularization
- Output layer with 3 classes (Home Win, Draw, Away Win)

## Performance

The model evaluates performance using:
- Overall accuracy
- Per-class precision, recall, and F1-score
- Confusion matrix visualization

## Acknowledgments

- Dataset source: https://www.kaggle.com/datasets/evangower/premier-league-matches-19922022

"""
EAS 574 Project 1: Chess Outcome Prediction
Training Script

Complete the TODO sections below. Read the docstrings carefully.

The data file contains:
- move_text_20: The first 20 moves of each game (for feature engineering)
- 21 baseline features (already extracted)
- 4 placeholder columns for YOUR features (custom_feature_1 through 4)
- outcome: Game result (0.0 = White loss, 0.5 = Draw, 1.0 = White win)

You should split the data into train/test sets yourself to evaluate locally.
After the deadline, your model will be scored on hidden data you've never seen.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import joblib
import os

# Configuration
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
DATA_FILE = os.path.join(DATA_DIR, 'chess_games.csv')
MODEL_FILE = os.path.join(os.path.dirname(__file__), '..', 'chess_model.pkl')
SCALER_FILE = os.path.join(os.path.dirname(__file__), '..', 'feature_scaler.pkl')

# All feature columns (21 baseline + 4 custom placeholders)
FEATURE_COLUMNS = [
    # Baseline features (21)
    'white_pawns', 'white_knights', 'white_bishops', 'white_rooks', 'white_queens',
    'black_pawns', 'black_knights', 'black_bishops', 'black_rooks', 'black_queens',
    'material_imbalance',
    'white_legal_moves', 'black_legal_moves',
    'white_kingside_castling', 'white_queenside_castling',
    'black_kingside_castling', 'black_queenside_castling',
    'white_advanced_pawns', 'black_advanced_pawns',
    'white_king_center_distance', 'black_king_center_distance',
    # YOUR custom features (4 placeholders - overwrite these!)
    'custom_feature_1',
    'custom_feature_2',
    'custom_feature_3',
    'custom_feature_4',
]

TARGET_COLUMN = 'outcome'


def add_custom_features(df):
    """
    Compute your custom features and overwrite the placeholder columns.

    The dataset includes 4 placeholder columns (custom_feature_1 through 4)
    that are set to zero. Your job is to overwrite them with meaningful
    features computed from move_text_20.
    """
    import chess

    # =========================================================================
    # Helper function to replay moves and get the board position
    # =========================================================================
    def get_board(move_text_20):
        """Replay moves and return the board position."""
        board = chess.Board()
        for token in move_text_20.split():
            if not token.endswith('.'):  # Skip move numbers like "1." "2."
                try:
                    board.push_san(token)
                except:
                    pass
        return board

    # =========================================================================
    # EXAMPLE FEATURE 1: Bishop pair advantage
    # Returns: 1 if White has both bishops, -1 if Black does, 0 otherwise
    # This is a WORKING example - you can keep it or replace it!
    # =========================================================================
    def bishop_pair(move_text_20):
        board = get_board(move_text_20)
        white_bishops = len(board.pieces(chess.BISHOP, chess.WHITE))
        black_bishops = len(board.pieces(chess.BISHOP, chess.BLACK))
        white_pair = 1 if white_bishops == 2 else 0
        black_pair = 1 if black_bishops == 2 else 0
        return white_pair - black_pair

    df['custom_feature_1'] = df['move_text_20'].apply(bishop_pair)

    # =========================================================================
    # TODO: Add 3 more features! Here are some ideas:
    # =========================================================================
    #
    # def center_control(move_text_20):
    #     """Count pieces attacking central squares (d4, d5, e4, e5)."""
    #     board = get_board(move_text_20)
    #     center = [chess.D4, chess.D5, chess.E4, chess.E5]
    #     white_control = sum(len(board.attackers(chess.WHITE, sq)) for sq in center)
    #     black_control = sum(len(board.attackers(chess.BLACK, sq)) for sq in center)
    #     return white_control - black_control
    #
    # df['custom_feature_2'] = df['move_text_20'].apply(center_control)
    #
    # df['custom_feature_3'] = df['move_text_20'].apply(your_function_3)
    # df['custom_feature_4'] = df['move_text_20'].apply(your_function_4)

    return df


def load_data(filepath):
    """
    Load data from CSV file.

    TODO:
        1. Load the CSV file using pandas
        2. Print the shape
        3. Return the DataFrame
    """
    # TODO: Implement this function
    pass


def prepare_features_and_labels(df, feature_columns):
    """
    Extract features (X) and labels (y) from the DataFrame.

    TODO:
        1. Extract feature columns as numpy array
        2. Convert outcomes to integer labels:
           - 0.0 -> 0 (White loss)
           - 0.5 -> 1 (Draw)
           - 1.0 -> 2 (White win)
        3. Return X and y
    """
    # TODO: Implement this function
    pass


def normalize_features(X):
    """
    Normalize features using StandardScaler.

    TODO:
        1. Create a StandardScaler
        2. Fit and transform X
        3. Return normalized X AND the scaler (you need it for prediction!)
    """
    # TODO: Implement this function
    pass


def train_model(X, y):
    """
    Train a logistic regression model.

    TODO:
        1. Create LogisticRegression with multi_class='multinomial', max_iter=1000
        2. Fit the model
        3. Print training accuracy and F1 score
        4. Return the model
    """
    # TODO: Implement this function
    pass


def save_artifacts(model, scaler, model_path, scaler_path):
    """
    Save model and scaler using joblib.

    TODO:
        1. Save model to model_path
        2. Save scaler to scaler_path
        3. Print confirmation
    """
    # TODO: Implement this function
    pass


def main():
    """Main training pipeline."""
    print("=" * 60)
    print("EAS 574 Project 1: Training")
    print("=" * 60)

    # Load data
    print(f"\nLoading data from {DATA_FILE}")
    df = load_data(DATA_FILE)

    # Add your custom features (overwrites placeholder columns)
    print("\nComputing custom features...")
    df = add_custom_features(df)

    # Prepare features and labels
    print("\nPreparing features and labels...")
    X, y = prepare_features_and_labels(df, FEATURE_COLUMNS)
    print(f"Features shape: {X.shape}")
    print(f"Using {len(FEATURE_COLUMNS)} features (21 baseline + 4 custom)")

    # Normalize
    print("\nNormalizing features...")
    X_norm, scaler = normalize_features(X)

    # Train
    print("\nTraining model...")
    model = train_model(X_norm, y)

    # Save
    print("\nSaving model and scaler...")
    save_artifacts(model, scaler, MODEL_FILE, SCALER_FILE)

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()

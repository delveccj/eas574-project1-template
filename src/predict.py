"""
EAS 574 Project 1: Chess Outcome Prediction
Prediction Interface

The grader calls the predict() function directly.
DO NOT change the function signature.

IMPORTANT: The grader passes the same columns as chess_games.csv
(move_text_20 + 21 baseline + 4 placeholders), but placeholders are zeros.
You must compute your custom features here too!
"""

import pandas as pd
import numpy as np
import joblib
import os

# Configuration
MODEL_FILE = os.path.join(os.path.dirname(__file__), '..', 'chess_model.pkl')
SCALER_FILE = os.path.join(os.path.dirname(__file__), '..', 'feature_scaler.pkl')
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

# All feature columns (must match train.py exactly!)
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
    # Custom features (4 placeholders)
    'custom_feature_1',
    'custom_feature_2',
    'custom_feature_3',
    'custom_feature_4',
]


def add_custom_features(df):
    """
    Compute your custom features - MUST MATCH train.py EXACTLY!

    Copy your feature computation code from train.py here.
    The grader passes zeros in the placeholder columns,
    so you must overwrite them with the same features you trained on.
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
    # EXAMPLE FEATURE 1: Bishop pair advantage (matches train.py)
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
    # TODO: Copy your other features from train.py here!
    # =========================================================================
    # df['custom_feature_2'] = df['move_text_20'].apply(your_function_2)
    # df['custom_feature_3'] = df['move_text_20'].apply(your_function_3)
    # df['custom_feature_4'] = df['move_text_20'].apply(your_function_4)

    return df


def predict(features_input):
    """
    Generate probability predictions for chess game outcomes.

    THIS IS THE GRADING CONTRACT - DO NOT CHANGE THE SIGNATURE

    Args:
        features_input: DataFrame with columns:
            - move_text_20 (string)
            - 21 baseline features
            - 4 placeholder columns (zeros - you must overwrite!)

    Returns:
        numpy array, shape (n_samples, 3)
        - Column 0: P(White loss)
        - Column 1: P(Draw)
        - Column 2: P(White win)
        - Each row sums to 1.0

    TODO:
        1. Convert input to DataFrame if needed
        2. Call add_custom_features() to compute your features
        3. Extract the feature columns as numpy array
        4. Load scaler and model
        5. Normalize with scaler.transform() (NOT fit_transform!)
        6. Get probabilities with model.predict_proba()
        7. Return probabilities
    """
    # TODO: Implement this function
    pass


def test_locally():
    """
    Test your predict function locally.
    Run with: python src/predict.py
    """
    print("=" * 60)
    print("Testing predict() locally")
    print("=" * 60)

    # Load data
    data_file = os.path.join(DATA_DIR, 'chess_games.csv')
    print(f"\nLoading {data_file}...")

    try:
        df = pd.read_csv(data_file)
        print(f"Loaded {len(df)} samples")
    except FileNotFoundError:
        print(f"ERROR: {data_file} not found")
        return

    # Simulate grader: take last 1000, drop outcome, zero out placeholders
    test_df = df.tail(1000).copy()
    test_df = test_df.drop(columns=['outcome'])
    test_df['custom_feature_1'] = 0
    test_df['custom_feature_2'] = 0
    test_df['custom_feature_3'] = 0
    test_df['custom_feature_4'] = 0

    print(f"Testing on {len(test_df)} samples (placeholders zeroed)...")

    # Get predictions
    print("\nCalling predict()...")
    try:
        probs = predict(test_df)
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return

    # Validate
    if probs is None:
        print("ERROR: predict() returned None")
        return

    if not isinstance(probs, np.ndarray):
        print(f"ERROR: Expected numpy array, got {type(probs)}")
        return

    if probs.shape != (len(test_df), 3):
        print(f"ERROR: Expected shape {(len(test_df), 3)}, got {probs.shape}")
        return

    if not np.allclose(probs.sum(axis=1), 1.0, atol=1e-5):
        print("ERROR: Rows don't sum to 1.0")
        return

    # Success
    print("\n" + "=" * 60)
    print("SUCCESS! Predictions are valid.")
    print("=" * 60)

    print(f"\nShape: {probs.shape}")
    print(f"\nPredicted class distribution:")
    pred_classes = np.argmax(probs, axis=1)
    for c, name in enumerate(['White loss', 'Draw', 'White win']):
        count = (pred_classes == c).sum()
        print(f"  {name}: {count} ({100*count/len(pred_classes):.1f}%)")


if __name__ == '__main__':
    test_locally()

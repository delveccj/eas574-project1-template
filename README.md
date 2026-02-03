# EAS 574 Project 1: Chess Outcome Prediction

Predict chess game outcomes (White win, Draw, White loss) from board positions at move 20.

## Quick Start

```bash
# 1. Create and activate virtual environment
python -m venv venv
source venv/bin/activate      # Mac/Linux
venv\Scripts\activate         # Windows

# 2. Install dependencies (after you create requirements.txt)
pip install -r requirements.txt

# 3. Train your model
python src/train.py

# 4. Test predictions locally
python src/predict.py

# 5. Push to trigger validation check
git add .
git commit -m "Complete implementation"
git push
```

## Project Structure

```
your-repo/
|-- README.md               # This file
|-- requirements.txt        # YOU CREATE THIS
|-- data/
|   +-- chess_games.csv     # 39K games with features
+-- src/
    |-- train.py            # Training script (complete TODOs)
    +-- predict.py          # Prediction interface (complete TODOs)
```

## Dataset

**File:** `data/chess_games.csv` (~39,000 games)

Each row contains:
- `move_text_20`: First 20 moves (use this for feature engineering!)
- 21 baseline features (material, activity, castling, pawns, king position)
- 4 placeholder columns (`custom_feature_1` through `custom_feature_4`) — **all zeros, for YOUR features**
- `outcome`: 0.0 (White loss), 0.5 (Draw), 1.0 (White win)

## Feature Engineering

The dataset has 4 placeholder columns set to zero. **Overwrite them** with your custom features:

```python
# In train.py and predict.py:
def add_custom_features(df):
    # Parse moves and compute features
    df['custom_feature_1'] = df['move_text_20'].apply(your_function_1)
    df['custom_feature_2'] = df['move_text_20'].apply(your_function_2)
    # ... etc
    return df
```

The grader passes zeros in these columns — your code must fill them in!

## Grading Contract

Your `predict()` function must:
- **Input:** DataFrame with move_text_20 + 21 baseline + 4 placeholders
- **Output:** numpy array, shape (n_samples, 3)
  - Column 0: P(White loss)
  - Column 1: P(Draw)
  - Column 2: P(White win)
  - Each row sums to 1.0

## Tips

- **Split your data** with `train_test_split()` to evaluate locally
- **Save your scaler** — you need it during prediction
- **Use `transform()` not `fit_transform()`** on test data
- **Copy your feature code** from train.py to predict.py exactly
- **Return probabilities** with `model.predict_proba()`, not `model.predict()`

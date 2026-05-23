"""Retrain noise filter on synthetic varied data for better demo output."""
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from xgboost import XGBClassifier
from noise_filter import FEATURE_COLS, MODEL_PATH

np.random.seed(42)
N = 1000

# Generate synthetic varied clashes
def synthetic_row(is_real: int):
    if is_real:
        return {
            "has_title": 1,
            "has_description": np.random.choice([0, 1], p=[0.2, 0.8]),
            "title_length": np.random.randint(20, 100),
            "desc_length": np.random.randint(50, 500),
            "desc_word_count": np.random.randint(10, 80),
            "desc_density": np.random.uniform(0.3, 0.9),
            "has_assigned_to": np.random.choice([0, 1], p=[0.3, 0.7]),
            "has_due_date": np.random.choice([0, 1], p=[0.4, 0.6]),
            "has_stage": np.random.choice([0, 1], p=[0.5, 0.5]),
            "has_bim_snippet": np.random.choice([0, 1], p=[0.6, 0.4]),
            "num_ref_links": np.random.randint(0, 5),
            "num_comments": np.random.randint(1, 10),
            "num_viewpoints": np.random.randint(1, 5),
            "num_labels": np.random.randint(0, 3),
            "num_related_topics": np.random.randint(0, 3),
            "avg_comment_length": np.random.randint(30, 200),
            "num_unique_commenters": np.random.randint(1, 5),
            "status_rank": np.random.randint(1, 5),
            "is_open": np.random.choice([0, 1]),
            "is_closed": np.random.choice([0, 1]),
            "is_in_progress": np.random.choice([0, 1]),
            "priority_rank": np.random.randint(1, 5),
            "days_since_created": np.random.randint(1, 365),
            "days_open": np.random.randint(0, 100),
            "has_been_modified": 1,
            "noise_keyword_hit": 0,
            "clash_keyword_hit": np.random.choice([0, 1], p=[0.3, 0.7]),
            "type_encoded": np.random.randint(0, 5),
            "status_encoded": np.random.randint(0, 4),
        }
    else:  # noise
        return {
            "has_title": np.random.choice([0, 1], p=[0.1, 0.9]),
            "has_description": np.random.choice([0, 1], p=[0.8, 0.2]),
            "title_length": np.random.randint(5, 30),
            "desc_length": np.random.randint(0, 50),
            "desc_word_count": np.random.randint(0, 8),
            "desc_density": np.random.uniform(0.0, 0.3),
            "has_assigned_to": np.random.choice([0, 1], p=[0.8, 0.2]),
            "has_due_date": np.random.choice([0, 1], p=[0.9, 0.1]),
            "has_stage": np.random.choice([0, 1], p=[0.8, 0.2]),
            "has_bim_snippet": 0,
            "num_ref_links": 0,
            "num_comments": np.random.randint(0, 2),
            "num_viewpoints": np.random.randint(0, 2),
            "num_labels": 0,
            "num_related_topics": 0,
            "avg_comment_length": np.random.randint(0, 30),
            "num_unique_commenters": np.random.randint(0, 2),
            "status_rank": np.random.randint(0, 3),
            "is_open": np.random.choice([0, 1]),
            "is_closed": 0,
            "is_in_progress": 0,
            "priority_rank": np.random.randint(0, 2),
            "days_since_created": np.random.randint(1, 30),
            "days_open": np.random.randint(0, 30),
            "has_been_modified": np.random.choice([0, 1]),
            "noise_keyword_hit": np.random.choice([0, 1], p=[0.5, 0.5]),
            "clash_keyword_hit": np.random.choice([0, 1], p=[0.7, 0.3]),
            "type_encoded": np.random.randint(0, 5),
            "status_encoded": np.random.randint(0, 4),
        }

rows = []
labels = []
for _ in range(N // 2):
    rows.append(synthetic_row(1))
    labels.append(1)
    rows.append(synthetic_row(0))
    labels.append(0)

df = pd.DataFrame(rows)
X = df[FEATURE_COLS].values
y = np.array(labels)

model = XGBClassifier(
    n_estimators=200, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
    reg_alpha=0.1, reg_lambda=1.0, eval_metric="logloss",
    random_state=42, n_jobs=-1,
)
model.fit(X, y)

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)

print(f"Trained on {N} synthetic clashes, saved to {MODEL_PATH}")
print(f"Training accuracy: {model.score(X, y):.3f}")

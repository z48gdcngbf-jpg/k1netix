"""Retrain noise filter with graduated quality levels for varied demo output."""
import numpy as np
import pandas as pd
import pickle
from xgboost import XGBClassifier
from noise_filter import FEATURE_COLS, MODEL_PATH

np.random.seed(42)
N = 2000

def make_row(quality: float) -> dict:
    q = quality
    return {
        "has_title": int(np.random.random() < (0.5 + 0.5*q)),
        "has_description": int(np.random.random() < q),
        "title_length": int(5 + 95*q + np.random.normal(0, 10)),
        "desc_length": int(500*q + np.random.normal(0, 30)),
        "desc_word_count": int(80*q + np.random.normal(0, 5)),
        "desc_density": np.clip(q + np.random.normal(0, 0.1), 0, 1),
        "has_assigned_to": int(np.random.random() < q),
        "has_due_date": int(np.random.random() < q*0.8),
        "has_stage": int(np.random.random() < q*0.6),
        "has_bim_snippet": int(np.random.random() < q*0.5),
        "num_ref_links": int(5*q + np.random.normal(0, 1)),
        "num_comments": int(10*q + np.random.normal(0, 2)),
        "num_viewpoints": int(5*q + np.random.normal(0, 1)),
        "num_labels": int(3*q + np.random.normal(0, 1)),
        "num_related_topics": int(3*q + np.random.normal(0, 1)),
        "avg_comment_length": int(200*q + np.random.normal(0, 20)),
        "num_unique_commenters": int(5*q + np.random.normal(0, 1)),
        "status_rank": np.random.randint(0, 5),
        "is_open": np.random.choice([0, 1]),
        "is_closed": np.random.choice([0, 1]),
        "is_in_progress": np.random.choice([0, 1]),
        "priority_rank": int(4*q + np.random.normal(0, 1)),
        "days_since_created": np.random.randint(1, 365),
        "days_open": np.random.randint(0, 100),
        "has_been_modified": int(np.random.random() < (0.3 + 0.7*q)),
        "noise_keyword_hit": int(np.random.random() < (0.7 - 0.6*q)),
        "clash_keyword_hit": int(np.random.random() < (0.2 + 0.7*q)),
        "type_encoded": np.random.randint(0, 5),
        "status_encoded": np.random.randint(0, 4),
    }

rows, labels = [], []
for _ in range(N):
    quality = np.clip(np.random.beta(2, 2), 0, 1)
    row = make_row(quality)
    label = 1 if quality > 0.5 else 0
    rows.append(row)
    labels.append(label)

df = pd.DataFrame(rows)
for col in FEATURE_COLS:
    df[col] = df[col].clip(lower=0)

X = df[FEATURE_COLS].values
y = np.array(labels)

model = XGBClassifier(
    n_estimators=300, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
    reg_alpha=0.1, reg_lambda=1.0, eval_metric="logloss",
    random_state=42, n_jobs=-1,
)
model.fit(X, y)

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)

print(f"Trained on {N} graduated synthetic clashes")
print(f"Training accuracy: {model.score(X, y):.3f}")

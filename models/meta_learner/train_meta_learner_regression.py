import math
import sqlite3

import lightgbm as lgb
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from models.meta_learner.export_to_js import export_model_to_js
from util import root_path

META_DB = root_path() / "data/meta_dataset.db"
MODEL_SAVE_PATH = root_path() / "data/lgbm_meta_model_rmse.txt"


def train_lightgbm():
    print("[Train] Loading dataset from SQLite...")
    conn = sqlite3.connect(META_DB)
    df = pd.read_sql_query("SELECT * FROM meta_train", conn)
    conn.close()

    if df.empty:
        raise ValueError("Dataset is empty. Run prepare_meta_data.py first.")

    feature_names = [
        "User_Rating_Count", "User_Avg", "User_Std",
        "Movie_Rating_Count", "Movie_Avg", "Movie_Std", "Release_Year",

        "UserKNN_RMSE_Score", "UserKNN_Hit_Score",
        "ItemKNN_RMSE_Score", "ItemKNN_Hit_Score",
        "ContentKNN_RMSE_Score", "ContentKNN_Hit_Score",
        "SVD_Score", "AutoRec_Score",
    ]

    X = df[feature_names]
    y = df["Actual_Rating"]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    print("[Train] Training LightGBM Gradient Boosting Tree...")
    model = lgb.LGBMRegressor(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train, eval_set=[(X_val, y_val)])

    y_pred = model.predict(X_val)
    final_rmse = math.sqrt(mean_squared_error(y_val, y_pred))
    print("\n" + "=" * 50)
    print(f" LightGBM Meta-Learner RMSE: {final_rmse:.4f}")
    print("=" * 50)

    print("[Train] Generating Feature Importance Plot...")
    plt.figure(figsize=(10, 6))
    lgb.plot_importance(model, importance_type='gain', title='Meta-Learner Feature Importance (Information Gain)',
                        xlabel='Gain', max_num_features=12)
    plt.tight_layout()
    plt.savefig('lgbm_feature_importance_rmse.png', dpi=300)

    print(f"[Train] Saving Native LightGBM model to {MODEL_SAVE_PATH}...")
    model.booster_.save_model(str(MODEL_SAVE_PATH))
    export_model_to_js(model,root_path() / "webui/lgbm_rmse.js")
    print("[Train] Done!")


if __name__ == "__main__":
    train_lightgbm()

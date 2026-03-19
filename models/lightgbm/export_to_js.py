import joblib
import m2cgen as m2c
from util import root_path

MODEL_SAVE_PATH = root_path() / "data/lgbm_meta_model.pkl"
JS_EXPORT_PATH = root_path() / "data/lgbm_ranker.js"


def export_model_to_js():
    print(f"[Export] Loading LightGBM model from {MODEL_SAVE_PATH}...")
    try:
        model = joblib.load(MODEL_SAVE_PATH)
    except FileNotFoundError:
        raise Exception("Model file not found. Please run train_meta_learner.py first.")

    print("[Export] Compiling LightGBM to Pure Native JavaScript...")
    js_code = m2c.export_to_javascript(model)

    wrapped_js = f"""// Auto-generated LightGBM Meta-Learner for CineMatch Edge AI
// Required Feature Array Order:
// [User_Rating_Count, User_Avg, User_Std, Movie_Rating_Count, Movie_Avg, Movie_Std, Release_Year, AutoRec, UserKNN, ItemKNN, SVD, ContentKNN]

{js_code}

// Main interface for frontend
function rankMovie(featuresArray) {{
    if (featuresArray.length !== 12) {{
        console.error("Expected exactly 12 features.");
        return 0;
    }}
    return score(featuresArray);
}}
"""
    with open(JS_EXPORT_PATH, "w") as f:
        f.write(wrapped_js)

    print(f"[Export] Compilation successful! Saved to {JS_EXPORT_PATH}.")


if __name__ == "__main__":
    export_model_to_js()
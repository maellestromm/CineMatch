import m2cgen as m2c


def export_model_to_js(model,path):
    print("[Export] Compiling LightGBM Booster to Pure Native JavaScript...")
    js_code = m2c.export_to_javascript(model)

    # 3. 包装给前端调用
    wrapped_js = f"""// Auto-generated LightGBM Meta-Learner for CineMatch Edge AI
// Required Feature Array Order (15 dimensions):
// [User_Rating_Count, User_Avg, User_Std, 
//  Movie_Rating_Count, Movie_Avg, Movie_Std, Release_Year, 
//  UserKNN_RMSE, UserKNN_Hit, ItemKNN_RMSE, ItemKNN_Hit, 
//  ContentKNN_RMSE, ContentKNN_Hit, SVD, AutoRec]

{js_code}

// Main interface for frontend
function rankMovie(featuresArray) {{
    if (featuresArray.length !== 15) {{
        console.error("Expected exactly 15 features, got " + featuresArray.length);
        return 0;
    }}

    let rawScore = score(featuresArray);

    if (Array.isArray(rawScore)) {{
        return rawScore[rawScore.length - 1]; 
    }}

    return rawScore;
}}
"""
    with open(path, "w") as f:
        f.write(wrapped_js)

    print(f"[Export] Compilation successful! Saved to {path}.")
import numpy as np
import pickle
from util import root_path

SAVE_DIR = root_path() / "models/saved_models"

# loads and returns a model given the name of a model pkl file
def load_model(pkl_name):
    file_name = pkl_name + ".pkl"
    with open(SAVE_DIR / file_name, "rb") as f:
        model = pickle.load(f)
    return model

# loads and returns all models as a dict structure
# models = {"name": model}
def load_all_models():
    models = {
        "User-KNN": load_model("user_knn"),
        "Deep-AutoRec": load_model("auto_rec"),
        "Item-KNN": load_model("item_knn"),
        "SVD-50": load_model("svd"),
        "Content-KNN": load_model("content_knn")
    }
    return models
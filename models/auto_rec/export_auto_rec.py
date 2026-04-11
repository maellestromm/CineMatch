import json
import os

import torch

from infer_autorec import DeepAutoRec
from util import root_path, merge_onnx_model


def export_model_to_frontend():
    print("[ONNX Exporter] loading model...")

    dict_path = root_path() / "data/movie_dictionary.json"
    weights_path = root_path() / "data/autorec_best_weights.pth"

    web_dir = root_path() / "webui"
    os.makedirs(web_dir, exist_ok=True)
    onnx_export_path = web_dir / "autorec.onnx"

    with open(dict_path, "r", encoding="utf-8") as f:
        movie_slugs = json.load(f)
    num_movies = len(movie_slugs)

    model = DeepAutoRec(num_movies=num_movies)
    model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))

    model.eval()

    dummy_input = torch.zeros(1, num_movies)

    print("[ONNX Exporter] exporting onnx...")
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_export_path),
        export_params=True,
        do_constant_folding=True,
        input_names=['user_ratings'],
        output_names=['predictions']
    )

    print(f"The ONNX model has been saved to: {onnx_export_path}")


if __name__ == "__main__":
    export_model_to_frontend()
    web_dir = root_path() / "webui"
    onnx_path = web_dir / "autorec.onnx"
    data_path = web_dir / "autorec.onnx.data"
    merge_onnx_model(onnx_path, data_path)

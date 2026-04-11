import torch

from nn_recommender import WideAndDeepMeta
from util import root_path, merge_onnx_model


def export_nn_onnx():
    model_path = root_path() / "data/nn_meta_model.pth"
    onnx_path = root_path() / "webui/nn_meta.onnx"

    print("[Export] loading model...")
    model = WideAndDeepMeta(input_dim=5)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    dummy_input = torch.randn(1, 5)

    print("[Export] exporting onnx...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        input_names=['features'],
        output_names=['score'],
        dynamic_axes={'features': {0: 'batch_size'}, 'score': {0: 'batch_size'}}
    )

    print(f"[Export] The ONNX model has been saved to: {onnx_path}")

if __name__ == "__main__":
    export_nn_onnx()
    web_dir = root_path() / "webui"
    onnx_path = web_dir / "nn_meta.onnx"
    data_path = web_dir / "nn_meta.onnx.data"
    merge_onnx_model(onnx_path, data_path)
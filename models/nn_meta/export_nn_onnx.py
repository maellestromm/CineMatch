import os

import onnx
import torch
from util import root_path, merge_onnx_model
from nn_recommender import WideAndDeepMeta


def export_nn_onnx():
    model_path = root_path() / "data/nn_meta_model.pth"
    onnx_path = root_path() / "webui/nn_meta.onnx"

    print("[Export] 加载 Wide & Deep 模型...")
    model = WideAndDeepMeta(input_dim=5)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    # 创建一个 dummy input，Batch Size 设为 1，特征数为 5
    dummy_input = torch.randn(1, 5)

    print("[Export] 正在导出为 ONNX 格式...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        input_names=['features'],  # 指定输入节点名称
        output_names=['score'],  # 指定输出节点名称
        # 🚀 极其关键：允许第一维 (Batch Size) 是动态的！
        # 这样前端就可以一次性传入 [3334, 5] 的矩阵，而不是只能传 [1, 5]
        dynamic_axes={'features': {0: 'batch_size'}, 'score': {0: 'batch_size'}}
    )

    print(f"[Export] 导出成功！ONNX 模型已保存至: {onnx_path}")

if __name__ == "__main__":
    export_nn_onnx()
    web_dir = root_path() / "webui"
    onnx_path = web_dir / "nn_meta.onnx"
    data_path = web_dir / "nn_meta.onnx.data"
    merge_onnx_model(onnx_path, data_path)
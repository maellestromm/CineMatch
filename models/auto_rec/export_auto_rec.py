import json
import os

import onnx
import torch

from infer_autorec import DeepAutoRec  # 直接导入你写好的网络结构
from util import root_path, merge_onnx_model


def export_model_to_frontend():
    print("[ONNX Exporter] 启动前端模型脱水程序...")

    # 路径配置
    dict_path = root_path() / "data/movie_dictionary.json"
    weights_path = root_path() / "data/autorec_best_weights.pth"

    # 假设你的前端文件都在 webui 目录下
    web_dir = root_path() / "webui"
    os.makedirs(web_dir, exist_ok=True)
    onnx_export_path = web_dir / "autorec.onnx"

    # 1. 加载电影字典，获取输入维度
    with open(dict_path, "r", encoding="utf-8") as f:
        movie_slugs = json.load(f)
    num_movies = len(movie_slugs)
    print(f"[ONNX Exporter] 侦测到电影总数 (输入维度): {num_movies}")

    # 2. 初始化 PyTorch 模型并加载你的最佳权重
    model = DeepAutoRec(num_movies=num_movies)
    model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))

    # 【极度关键】：必须切换到 eval 模式，否则前端推理时 Dropout 还会随机丢弃神经元！
    model.eval()

    # 3. 构造 Dummy Input (占位符张量)
    # 形状为 [1, num_movies]，代表 1 个用户的数据
    dummy_input = torch.zeros(1, num_movies)

    # 4. 执行 ONNX 静态图追踪与导出
    print("[ONNX Exporter] 正在追踪计算图并融合常量...")
    torch.onnx.export(
        model,  # 要导出的模型
        dummy_input,  # 样例输入
        str(onnx_export_path),  # 导出路径
        export_params=True,  # 将权重固化进模型中
        do_constant_folding=True,  # 开启常量折叠优化，减小文件体积
        input_names=['user_ratings'],  # 对应前端喂入数据的 Key
        output_names=['predictions']  # 对应前端提取结果的 Key
    )

    print("\n" + "=" * 50)
    print(f"🚀 大功告成！")
    print(f"1. 神经网络已编译为跨平台格式: {onnx_export_path}")
    print(f"2. 请务必将 data/movie_dictionary.json 也复制到前端目录。")
    print("=" * 50)

if __name__ == "__main__":
    export_model_to_frontend()
    web_dir = root_path() / "webui"
    onnx_path = web_dir / "autorec.onnx"
    data_path = web_dir / "autorec.onnx.data"
    merge_onnx_model(onnx_path, data_path)

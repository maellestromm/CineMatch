# CineMatch

## Requirements

pandas~=3.0.1\
scikit-learn~=1.8.0\
letterboxdpy~=6.4.1\
torch~=2.10.0\
matplotlib~=3.10.8\
numpy~=2.4.2\
scipy~=1.17.1.0

## Project Structure

├── db_backup/ # 数据库备份文件 (原始爬取数据)\
├── data/ # 运行时数据、模型权重、生成的字典等\
├── models/\
│ ├── content_knn/ # 基于内容的推荐 (Content-Based)\
│ ├── item_knn/ # 基于物品的协同过滤 (Item-KNN)\
│ ├── svd/ # 隐语义模型/矩阵分解 (Truncated SVD)\
│ ├── user_knn/ # 基于用户的协同过滤 (User-KNN)\
│ └── auto_rec/ # 深度学习自编码器 (Deep AutoRec)\
├── tools/\
│ ├── clear_db.py # 数据库清洗与预处理脚本\
│ ├── split_db.py # 训练集/测试集物理拆分脚本\
│ ├── Movie_first_crawler.py # 电影优先爬虫\
│ └── User_first_crawler.py # 用户优先爬虫\
├── util.py # 全局辅助与路径配置函数\
└── README.md # 项目文档

## 模型 (Recommendation Models)

本项目实现了五种横跨不同时代的经典推荐算法，以构建完整的召回与精排架构：

### 1. 内容 KNN (Content-KNN)

* **原理**：“这些电影与你之前看过的电影有相同标签。”
* **实现**：提取电影的导演、类型、演员、简介等元数据，构建 TF-IDF 文本特征向量，通过计算余弦相似度，找出与用户历史高分电影在物理属性上最接近的候选电影。解决了新电影的冷启动问题。

### 2. 物品 KNN (Item-KNN)

* **原理**：“喜欢这部电影的人，通常也都喜欢那部电影。”
* **实现**：基于全局的 Item-User 交互矩阵，计算物品与物品之间的余弦相似度。引入了贝叶斯平滑（Bayesian
  Smoothing）来惩罚小样本高分偏差。推理时通过张量矩阵乘法实现极速推荐。

### 3. 用户 KNN (User-KNN)

* **原理**：“品味和你相似的灵魂伴侣，他们看了什么。”
* **实现**：基于 User-Item 交互矩阵，寻找与目标用户打分趋势最接近的 K 个邻居。同样引入了先验均值与阻尼系数来消除热门偏见。支持
  CPU (scikit-learn) 与 GPU (PyTorch 张量计算) 双引擎后端。

### 4. 矩阵分解 (SVD-50)

* **原理**：“你不懂自己的品味，但数学懂。”
* **实现**：使用截断奇异值分解 (Truncated SVD) 将庞大的评分矩阵降维，提取出 50 个隐藏的语义维度（Latent Factors）。在推理阶段使用
  Folding-in（折入投影）技术，无需重新训练即可实现极低延迟的新用户评分预测。

### 5. 深度自编码器 (Deep AutoRec)

* **原理**：“用神经网络学习高维特征的非线性压缩与重建。”
* **实现**：构建了多层 Encoder-Decoder 架构，并引入 30% Dropout 防止过拟合。它能够极其精准地捕捉用户评分数据中复杂的非线性隐式关联。

## 爬虫 (Crawlers)

为了获取高质量的真实打分数据，我们设计了两套基于队列 (Queue) 交替抓取的互补爬虫策略：

* **用户优先爬虫 (User-First Crawler)**
    1. 获取用户队列 (User Queue) 中所有用户评论过的所有电影。
    2. 挑选出评论数最多的电影，获取其详细元数据，并提取该电影下的热门评论，将这些评论的作者加入用户队列。

    * **作用**：能够快速聚集在核心电影上产生过交互的用户群体，迅速提高 User-Item 矩阵的局部密度。

* **电影优先爬虫 (Movie-First Crawler)**
    1. 获取用户队列 (User Queue) 中某位用户评论过的所有电影，并将这些电影加入电影队列 (Movie Queue)。
    2. 获取电影队列中所有电影的详细元数据，并提取这些电影下的热门评论，将评论的作者加入用户队列。

    * **作用**：以探索电影的多样性为主，能够广泛拓宽数据库中的电影种类边界，为 Content-KNN 提供丰富的特征素材。

## 如何运行评测 (How to Run Benchmarks)

1. 解压 `db_backup/user_first_cut3_clear.7z` 到 `data/` 文件夹。
2. 运行 `tools/split_db.py`。这会将数据库按 9:1 的比例进行严格的物理拆分，生成 `train_model.db` 和 `test_eval.db`
   ，确保评估时零数据泄露。
3. （可选）进入 `models/auto_rec/` 运行 `train_autorec.py` 预训练深度学习模型。
4. 运行 `evaluate_strict.py` 查看各大模型在 Hit Rate（命中率）与 Precision 上的排位赛。
5. 运行 `evaluate_rmse.py` 查看各大模型在 1-5 星真实品味预测精度上的排位赛。

## 评测表现与结论 (Performance)

### 1. 召回与命中率测试 (Hit Rate & Precision @ 10)

测试目标：在茫茫片海中，能否盲猜中测试集中被隐去的真实交互电影？

| Model Name   | Hit Rate (@10) | Precision (@10) | Avg Latency |
|:-------------|:---------------|:----------------|:------------|
| **SVD-50**   | **72.92%**     | **25.04%**      | **9.3 ms**  |
| User-KNN     | 71.13%         | 18.06%          | 139.1 ms    |
| Item-KNN     | 57.91%         | 13.05%          | 167.5 ms    |
| Deep AutoRec | 46.49%         | 8.61%           | 2.5 ms      |
| Content-KNN  | 20.88%         | 2.95%           | 5.8 ms      |

### 2. 真实品味预测精度赛 (RMSE Score)

测试目标：已知用户看了一部电影，能否精准预测其 1-5 星的具体打分？(分数越低越好)

| Model Name       | RMSE Score |
|:-----------------|:-----------|
| **Deep AutoRec** | **0.7524** |
| User-KNN         | 0.8092     |
| SVD-50           | 0.8513     |
| Item-KNN         | 0.8875     |
| Content-KNN      | 0.9483     |


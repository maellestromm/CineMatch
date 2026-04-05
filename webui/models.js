import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.mjs';

export class AutoRec {
    constructor() {
        this.session = null;
        this.movieSlugs = [];
        this.movieToIdx = {};
        this.numMovies = 0;
    }

    /**
     * 初始化：加载 ONNX 模型和顺序字典
     */
    async initialize(
        modelUrl = './autorec.onnx',
        dictUrl = './movie_dictionary.json'
    ) {
        console.log("[AutoRec] 正在加载模型字典与权重...");
        try {
            const dictResponse = await fetch(dictUrl);
            this.movieSlugs = await dictResponse.json();
            this.numMovies = this.movieSlugs.length;

            // 建立 O(1) 的电影到索引映射
            this.movieSlugs.forEach((slug, idx) => {
                this.movieToIdx[slug] = idx;
            });

            // 唤醒 WebAssembly 神经网络引擎
            this.session = await ort.InferenceSession.create(modelUrl);
            console.log(`[AutoRec] 引擎就绪！已加载 ${this.numMovies} 部电影输入特征。`);
        } catch (error) {
            console.error("[AutoRec] 初始化失败:", error);
            throw error;
        }
    }

    /**
     * 全量预测：直接返回包含 3334 部电影预测分的字典
     * @param {Object} user_profile - 例如 {'inception': 5.0, 'interstellar': 4.5}
     * @returns {Promise<Object>} - 格式: { 'slug1': 4.2, 'slug2': 3.8, ... }
     */
    async get_recommendations(user_profile) {
        if (!this.session) {
            console.warn("[AutoRec] 警告：模型未初始化，请先调用 initialize()。");
            return {};
        }

        // 1. 构造全零张量向量
        const targetVector = new Float32Array(this.numMovies);
        const watchedIndices = [];
        let hasInput = false;

        // 2. 填入用户已知评分
        for (const [slug, rating] of Object.entries(user_profile)) {
            if (this.movieToIdx[slug] !== undefined) {
                const idx = this.movieToIdx[slug];
                targetVector[idx] = parseFloat(rating);
                watchedIndices.push(idx);
                hasInput = true;
            }
        }

        if (!hasInput) {
            console.warn("[AutoRec] 警告：用户输入的电影不在特征库中。");
            return {};
        }

        // 3. 构建张量并进行前向传播
        const tensor = new ort.Tensor('float32', targetVector, [1, this.numMovies]);
        const feeds = {user_ratings: tensor}; // key 必须和导出时的 input_names 一致
        const results = await this.session.run(feeds);

        // 4. 提取底层输出的一维数组
        const predictions = results.predictions.data;

        // 5. 将看过的电影强行置为极低分，防止重复推荐
        for (const idx of watchedIndices) {
            predictions[idx] = -999.0;
        }

        // 6. [核心修改]：不再排序切片，直接打包成 { slug: score } 字典返回
        const allScores = {};
        for (let i = 0; i < this.numMovies; i++) {
            allScores[this.movieSlugs[i]] = predictions[i];
        }

        return allScores;
    }
}

export class ContentKNN_Hit {
    constructor() {
        this.movieSlugs = [];
        this.movieToIdx = {};
        this.numMovies = 0;
        this.simMatrix = {}; // K=1 的极简映射表
    }

    /**
     * 初始化加载
     */
    async initialize(
        dictUrl = './movie_dictionary.json',
        simUrl = './content_knn_k1.json' // 加载 K=1 版本
    ) {
        console.log("[ContentKNN-Hit] 正在加载模型字典与 K=1 拓扑...");
        try {
            const dictResponse = await fetch(dictUrl);
            this.movieSlugs = await dictResponse.json();
            this.numMovies = this.movieSlugs.length;

            this.movieSlugs.forEach((slug, idx) => {
                this.movieToIdx[slug] = idx;
            });

            const simResponse = await fetch(simUrl);
            this.simMatrix = await simResponse.json();

            console.log(`[ContentKNN-Hit] 引擎就绪！已加载 O(1) 极速映射网络。`);
        } catch (error) {
            console.error("[ContentKNN-Hit] 初始化失败:", error);
            throw error;
        }
    }

    /**
     * 极速全量推断
     */
    async get_recommendations(user_profile) {
        if (!this.numMovies) return {};

        const targetVector = new Float32Array(this.numMovies);
        const hasRatedMask = new Float32Array(this.numMovies);
        const watchedIndices = [];

        let ratingsSum = 0;
        let userRatingsCount = 0;

        for (const [slug, rating] of Object.entries(user_profile)) {
            if (this.movieToIdx[slug] !== undefined) {
                const idx = this.movieToIdx[slug];
                const r = parseFloat(rating);
                targetVector[idx] = r;
                hasRatedMask[idx] = 1.0;
                watchedIndices.push(idx);

                ratingsSum += r;
                userRatingsCount += 1;
            }
        }

        if (userRatingsCount === 0) return {};

        const prior_mean = userRatingsCount > 0 ? (ratingsSum / userRatingsCount) : 3.0;
        const damping = 0.1;
        const allScores = {};

        // 🚀 O(N) 极速打分：每部电影只需 O(1) 查表
        for (let i = 0; i < this.numMovies; i++) {
            // 已看电影直接熔断
            if (watchedIndices.includes(i)) {
                allScores[this.movieSlugs[i]] = -999.0;
                continue;
            }

            const neighborData = this.simMatrix[i];
            let finalScore = 0.0;

            if (neighborData) {
                // 解构提取唯一的 [邻居索引, 相似度]
                const [bestJ, sim] = neighborData;

                // 判断用户是否看过这个“唯一邻居”
                if (hasRatedMask[bestJ] > 0) {
                    const recScore = sim * targetVector[bestJ];
                    if (sim >= 0.01) { // 噪音熔断
                        finalScore = (recScore + damping * prior_mean) / (sim + damping);
                    } else {
                        finalScore = prior_mean; // 相似度太低，退化为均值
                    }
                } else {
                    // 用户没看过平替，退化为均值
                    finalScore = prior_mean;
                }
            } else {
                // 孤岛电影，退化为均值
                finalScore = prior_mean;
            }

            allScores[this.movieSlugs[i]] = finalScore;
        }

        return allScores;
    }
}

export class ItemKNN_Hit {
    constructor() {
        this.movieSlugs = [];
        this.movieToIdx = {};
        this.numMovies = 0;
        this.simMatrix = {}; // 存储 K=7 稀疏映射表
    }

    /**
     * 初始化加载
     */
    async initialize(
        dictUrl = './movie_dictionary.json',
        simUrl = './item_knn_k7.json' // 加载 K=7 的协同过滤矩阵
    ) {
        console.log("[ItemKNN-Hit] 正在加载模型字典与 K=7 拓扑...");
        try {
            const dictResponse = await fetch(dictUrl);
            this.movieSlugs = await dictResponse.json();
            this.numMovies = this.movieSlugs.length;

            this.movieSlugs.forEach((slug, idx) => {
                this.movieToIdx[slug] = idx;
            });

            const simResponse = await fetch(simUrl);
            this.simMatrix = await simResponse.json();

            console.log(`[ItemKNN-Hit] 引擎就绪！已加载 O(K) 极速映射网络。`);
        } catch (error) {
            console.error("[ItemKNN-Hit] 初始化失败:", error);
            throw error;
        }
    }

    /**
     * 极速全量推断
     */
    async get_recommendations(user_profile) {
        if (!this.numMovies) return {};

        const targetVector = new Float32Array(this.numMovies);
        const hasRatedMask = new Float32Array(this.numMovies);
        const watchedIndices = [];

        let ratingsSum = 0;
        let userRatingsCount = 0;

        // 1. 将用户的历史打分向量化
        for (const [slug, rating] of Object.entries(user_profile)) {
            if (this.movieToIdx[slug] !== undefined) {
                const idx = this.movieToIdx[slug];
                const r = parseFloat(rating);
                targetVector[idx] = r;
                hasRatedMask[idx] = 1.0;
                watchedIndices.push(idx);

                ratingsSum += r;
                userRatingsCount += 1;
            }
        }

        if (userRatingsCount === 0) return {};

        // 计算用户的平均分，作为缺失预测的兜底 (Prior Mean)
        const prior_mean = userRatingsCount > 0 ? (ratingsSum / userRatingsCount) : 3.0;
        const damping = 0.1;
        const allScores = {};

        // 2. 核心预测逻辑
        for (let i = 0; i < this.numMovies; i++) {
            // 已看电影直接熔断
            if (watchedIndices.includes(i)) {
                allScores[this.movieSlugs[i]] = -999.0;
                continue;
            }

            const neighbors = this.simMatrix[i];
            let finalScore = prior_mean; // 默认使用用户均分兜底

            if (neighbors && neighbors.length > 0) {
                let recScore = 0.0;
                let simSum = 0.0;

                // 遍历该电影最多 7 个最相似的邻居
                for (let k = 0; k < neighbors.length; k++) {
                    const bestJ = neighbors[k][0];
                    const sim = neighbors[k][1];

                    // 只有当用户看过这个邻居时，它才能提供有效分数
                    if (hasRatedMask[bestJ] > 0) {
                        recScore += sim * targetVector[bestJ];
                        simSum += sim;
                    }
                }

                // 噪音熔断机制：累加相似度足够大时，才采用协同过滤分数
                if (simSum >= 0.01) {
                    finalScore = (recScore + damping * prior_mean) / (simSum + damping);
                }
            }

            allScores[this.movieSlugs[i]] = finalScore;
        }

        return allScores;
    }
}

export class SVDRecommender {
    constructor() {
        this.movieSlugs = [];
        this.movieToIdx = {};
        this.numMovies = 0;
        this.vt = []; // 保存 k x N 的右奇异矩阵
        this.k_factors = 39;
    }

    /**
     * 初始化加载
     */
    async initialize(
        dictUrl = './movie_dictionary.json',
        svdUrl = './svd_k39_vt.json' // 加载导出的 Vt 矩阵
    ) {
        console.log("[SVD] 正在加载模型字典与隐式向量矩阵...");
        try {
            const dictResponse = await fetch(dictUrl);
            this.movieSlugs = await dictResponse.json();
            this.numMovies = this.movieSlugs.length;

            this.movieSlugs.forEach((slug, idx) => {
                this.movieToIdx[slug] = idx;
            });

            const svdResponse = await fetch(svdUrl);
            this.vt = await svdResponse.json();

            // 确保矩阵维度正确
            this.k_factors = this.vt.length;

            console.log(`[SVD] 引擎就绪！已加载 k=${this.k_factors} 维隐特征矩阵。`);
        } catch (error) {
            console.error("[SVD] 初始化失败:", error);
            throw error;
        }
    }

    /**
     * 全量推断
     */
    async get_recommendations(user_profile) {
        if (!this.numMovies || !this.vt.length) return {};

        const targetVector = new Float32Array(this.numMovies);
        const watchedIndices = [];

        let ratingsSum = 0;
        let userRatingsCount = 0;

        // 1. 解析用户向量
        for (const [slug, rating] of Object.entries(user_profile)) {
            if (this.movieToIdx[slug] !== undefined) {
                const idx = this.movieToIdx[slug];
                const r = parseFloat(rating);
                targetVector[idx] = r;
                watchedIndices.push(idx);

                ratingsSum += r;
                userRatingsCount += 1;
            }
        }

        if (userRatingsCount === 0) return {};

        // 2. 均值兜底与中心化 (Centering)
        const prior_mean = userRatingsCount > 0 ? (ratingsSum / userRatingsCount) : 3.0;
        const targetCentered = new Float32Array(this.numMovies);

        for (let i = 0; i < this.numMovies; i++) {
            if (targetVector[i] > 0) {
                targetCentered[i] = targetVector[i] - prior_mean;
            } else {
                targetCentered[i] = 0.0; // 未评价电影填入均值，减去均值后为 0
            }
        }

        // 3. 🚀 核心计算：(target_centered @ Vt.T) @ Vt
        // 步骤 3.1: 降维 -> (1 x N) @ (N x K) = (1 x K)
        const hiddenVector = new Float32Array(this.k_factors);
        for (let k = 0; k < this.k_factors; k++) {
            let sum = 0.0;
            const vtRow = this.vt[k]; // 取出 Vt 的第 k 行
            for (let i = 0; i < this.numMovies; i++) {
                sum += targetCentered[i] * vtRow[i];
            }
            hiddenVector[k] = sum;
        }

        // 步骤 3.2: 重构 -> (1 x K) @ (K x N) = (1 x N)
        const reconstructedScores = new Float32Array(this.numMovies);
        for (let i = 0; i < this.numMovies; i++) {
            let sum = 0.0;
            for (let k = 0; k < this.k_factors; k++) {
                sum += hiddenVector[k] * this.vt[k][i];
            }
            reconstructedScores[i] = sum;
        }

        // 4. 去中心化并生成最终结果
        const allScores = {};
        for (let i = 0; i < this.numMovies; i++) {
            if (watchedIndices.includes(i)) {
                allScores[this.movieSlugs[i]] = -999.0;
            } else {
                // 加上之前减掉的用户均值
                allScores[this.movieSlugs[i]] = reconstructedScores[i] + prior_mean;
            }
        }

        return allScores;
    }
}

export class UserKNN_Hit {
    constructor() {
        this.movieSlugs = [];
        this.movieToIdx = {};
        this.numMovies = 0;
        this.users = []; // 存储所有历史用户的稀疏画像
        this.k_neighbors = 13; // 寻找 13 个灵魂伴侣
    }

    /**
     * 初始化加载
     */
    async initialize(
        dictUrl = './movie_dictionary.json',
        usersUrl = './user_knn_k13.json' // 加载导出的用户画像
    ) {
        console.log("[UserKNN-Hit] 正在加载模型字典与全量用户画像...");
        try {
            const dictResponse = await fetch(dictUrl);
            this.movieSlugs = await dictResponse.json();
            this.numMovies = this.movieSlugs.length;

            this.movieSlugs.forEach((slug, idx) => {
                this.movieToIdx[slug] = idx;
            });

            const usersResponse = await fetch(usersUrl);
            this.users = await usersResponse.json();

            console.log(`[UserKNN-Hit] 引擎就绪！已部署 ${this.users.length} 个历史用户档案，准备实时寻亲。`);
        } catch (error) {
            console.error("[UserKNN-Hit] 初始化失败:", error);
            throw error;
        }
    }

    /**
     * 实时寻亲并全量推断
     */
    async get_recommendations(user_profile) {
        if (!this.numMovies || this.users.length === 0) return {};

        const targetVector = new Float32Array(this.numMovies);
        const watchedIndices = [];
        let targetNormSq = 0;
        let ratingsSum = 0;
        let userRatingsCount = 0;

        // 1. 构建当前用户的特征向量，并计算 L2 范数
        for (const [slug, rating] of Object.entries(user_profile)) {
            if (this.movieToIdx[slug] !== undefined) {
                const idx = this.movieToIdx[slug];
                const r = parseFloat(rating);
                targetVector[idx] = r;
                watchedIndices.push(idx);

                targetNormSq += r * r;
                ratingsSum += r;
                userRatingsCount += 1;
            }
        }

        if (userRatingsCount === 0) return {};

        const targetNorm = Math.sqrt(targetNormSq);
        const prior_mean = userRatingsCount > 0 ? (ratingsSum / userRatingsCount) : 3.0;

        // 2. 🚀 计算当前用户与所有历史用户的余弦相似度 (Cosine Similarity)
        const similarities = new Float32Array(this.users.length);
        for (let i = 0; i < this.users.length; i++) {
            const user = this.users[i];
            let dotProduct = 0.0;

            // 稀疏向量点积计算（只遍历对方看过的电影，速度极快）
            for (const jStr of Object.keys(user.ratings)) {
                const j = parseInt(jStr);
                if (targetVector[j] > 0) {
                    dotProduct += targetVector[j] * user.ratings[j];
                }
            }

            if (targetNorm > 0 && user.norm > 0) {
                similarities[i] = dotProduct / (targetNorm * user.norm);
            } else {
                similarities[i] = 0.0;
            }
        }

        // 3. 寻找最相似的 K=13 个灵魂伴侣 (相当于 torch.topk)
        // 创建一个索引数组用于排序
        const indices = Array.from({length: this.users.length}, (_, i) => i);
        indices.sort((a, b) => similarities[b] - similarities[a]); // 降序排列
        const topKIndices = indices.slice(0, this.k_neighbors);

        const recommendationScores = new Float32Array(this.numMovies);
        const similaritySums = new Float32Array(this.numMovies);

        // 4. 将这 13 个伴侣的电影品味加权整合给当前用户
        for (let k = 0; k < topKIndices.length; k++) {
            const userIdx = topKIndices[k];
            const sim = similarities[userIdx];

            // 如果相似度过低甚至为负，就不采纳该用户的意见
            if (sim <= 0) continue;

            const user = this.users[userIdx];

            for (const jStr of Object.keys(user.ratings)) {
                const j = parseInt(jStr);
                const rating = user.ratings[j];

                recommendationScores[j] += sim * rating;
                similaritySums[j] += sim;
            }
        }

        const damping = 3.0; // 对齐 GPU 后端代码中的贝叶斯阻尼系数
        const allScores = {};

        // 5. 应用贝叶斯平滑并过滤已看电影
        for (let i = 0; i < this.numMovies; i++) {
            if (watchedIndices.includes(i)) {
                allScores[this.movieSlugs[i]] = -999.0;
                continue;
            }

            allScores[this.movieSlugs[i]] = (recommendationScores[i] + damping * prior_mean) / (similaritySums[i] + damping);
        }

        return allScores;
    }
}


// 确保在你的 models.js 中包含了这5个基座类的定义
// import { AutoRec, SVDRecommender, ItemKNN_Hit, ContentKNN_Hit, UserKNN_Hit } from './...';

export class NN_Meta {
    constructor() {
        this.session = null;
        this.movieSlugs = [];
        this.numMovies = 0;
        this.STD_CLIP_LOWER = 0.1;

        // 🚀 1. 内部挂载所有基座模型，前端页面无需再关心它们
        this.svd = new SVDRecommender();
        this.itemKnn = new ItemKNN_Hit();
        this.autoRec = new AutoRec();
        this.contentKnn = new ContentKNN_Hit();
        this.userKnn = new UserKNN_Hit();
    }

    /**
     * 一键初始化：利用 Promise.all 瞬间并发加载 6 个 JSON/ONNX 文件
     */
    async initialize() {
        console.log("[NN-Meta] 正在并发启动 5 大基座模型与神经网络融合引擎...");
        try {
            // 设置静态资源的基准路径 (根据你的项目实际路径调整)
            const dictUrl = './movie_dictionary.json';

            // 🚀 并发请求网络，将冷启动时间压缩到极致
            await Promise.all([
                this.svd.initialize(dictUrl, './svd_k39_vt.json'),
                this.itemKnn.initialize(dictUrl, './item_knn_k7.json'),
                this.autoRec.initialize('./autorec.onnx', dictUrl), // 注意 AutoRec 的参数顺序
                this.contentKnn.initialize(dictUrl, './content_knn_k1.json'),
                this.userKnn.initialize(dictUrl, './user_knn_k13.json'),

                // 并发加载 Meta 模型自己的配置
                (async () => {
                    const dictResponse = await fetch(dictUrl);
                    this.movieSlugs = await dictResponse.json();
                    this.numMovies = this.movieSlugs.length;
                    this.session = await ort.InferenceSession.create('./nn_meta.onnx');
                })()
            ]);

            console.log(`[NN-Meta] 引擎点火完毕！所有底层参数已加载到内存。`);
        } catch (error) {
            console.error("[NN-Meta] 启动失败，请检查网络或文件路径:", error);
            throw error;
        }
    }

    _getMeanAndStd(arr) {
        let sum = 0;
        for (let i = 0; i < arr.length; i++) sum += arr[i];
        const mean = sum / arr.length;

        let varianceSum = 0;
        for (let i = 0; i < arr.length; i++) {
            varianceSum += (arr[i] - mean) * (arr[i] - mean);
        }
        let std = Math.sqrt(varianceSum / arr.length);
        std = Math.max(std, this.STD_CLIP_LOWER);

        return {mean, std};
    }

    /**
     * 黑盒推断：接收用户特征，直接返回排序好的最终列表
     */
    async get_recommendations(user_profile) {
        if (!this.session) {
            console.warn("[NN-Meta] 模型尚未初始化！");
            return [];
        }

        // 🚀 1. 内部并发执行 5 个基座模型的预测！
        // 因为它们都是纯 CPU/内存 运算，Promise.all 会在几毫秒内全部执行完毕
        const [svdPreds, itemKnnPreds, autoRecPreds, contentKnnPreds, userKnnPreds] = await Promise.all([
            this.svd.get_recommendations(user_profile),
            this.itemKnn.get_recommendations(user_profile),
            this.autoRec.get_recommendations(user_profile),
            this.contentKnn.get_recommendations(user_profile),
            this.userKnn.get_recommendations(user_profile)
        ]);

        const base_predictions = {
            "SVD": svdPreds,
            "ItemKNN": itemKnnPreds,
            "AutoRec": autoRecPreds,
            "ContentKNN": contentKnnPreds,
            "UserKNN": userKnnPreds
        };

        const userRatings = Object.values(user_profile).map(Number);
        const userStats = this._getMeanAndStd(userRatings);
        const user_avg = userStats.mean || 3.5;
        const user_std = userRatings.length > 1 ? userStats.std : 1.0;

        const modelNames = ["SVD", "ItemKNN", "AutoRec", "ContentKNN", "UserKNN"];
        const normalizedScores = {};

        // 🚀 2. 清爽的 Z-score 动态预处理
        for (const modelName of modelNames) {
            const preds = base_predictions[modelName];
            if (!preds) continue;

            // 使用 this.movieSlugs 确保顺序和数量绝对正确
            const scores = new Float32Array(this.numMovies);

            for (let i = 0; i < this.numMovies; i++) {
                const slug = this.movieSlugs[i];
                let s = preds[slug];

                // 🚨 终极修复：完美对齐 Python 的隐式剔除机制
                // 拦截基座模型吐出的已看标记(-999)、冷启动异常(<=0) 或 undefined
                // 统一用 user_avg 填补，防止它们严重污染全局方差 (STD)！
                if (s === undefined || s <= 0) {
                    s = user_avg;
                }

                scores[i] = s;
            }

            const {mean, std} = this._getMeanAndStd(scores);

            normalizedScores[modelName] = {};
            for (let i = 0; i < this.numMovies; i++) {
                const slug = this.movieSlugs[i];
                normalizedScores[modelName][slug] = (scores[i] - mean) / std;
            }
        }

        // 3. 构建 [N, 5] 的大张量用于 ONNX 批处理
        const batchedInput = new Float32Array(this.numMovies * 5);
        for (let i = 0; i < this.numMovies; i++) {
            const slug = this.movieSlugs[i];
            const offset = i * 5;
            batchedInput[offset + 0] = (normalizedScores["SVD"] && normalizedScores["SVD"][slug]) || 0.0;
            batchedInput[offset + 1] = (normalizedScores["ItemKNN"] && normalizedScores["ItemKNN"][slug]) || 0.0;
            batchedInput[offset + 2] = (normalizedScores["AutoRec"] && normalizedScores["AutoRec"][slug]) || 0.0;
            batchedInput[offset + 3] = (normalizedScores["ContentKNN"] && normalizedScores["ContentKNN"][slug]) || 0.0;
            batchedInput[offset + 4] = (normalizedScores["UserKNN"] && normalizedScores["UserKNN"][slug]) || 0.0;
        }

        // 4. 一次性通过 WebAssembly 执行前向传播
        const tensor = new ort.Tensor('float32', batchedInput, [this.numMovies, 5]);
        const results = await this.session.run({features: tensor});
        const nnScores = results.score.data;

        // 🚀 修改：我们返回一个以 slug 为键的对象，方便前端快速查找并更新 UI
        const finalResults = {};

        // 5. 后处理：绝对分还原、组装 6 大模型分数，并保留已看电影
        for (let i = 0; i < this.numMovies; i++) {
            const slug = this.movieSlugs[i];

            let absoluteScore;
            let isWatched = false;

            // 🚀 核心逻辑修改：如果是看过的电影，直接使用用户的真实打分！
            if (user_profile[slug] !== undefined) {
                absoluteScore = parseFloat(user_profile[slug]);
                isWatched = true;
            } else {
                // 没看过的电影，使用 NN 预测并还原绝对分
                absoluteScore = (nnScores[i] * user_std) + user_avg;
                absoluteScore = absoluteScore;
            }

            finalResults[slug] = {
                meta: absoluteScore,
                svd: base_predictions["SVD"][slug] || 0,
                itemknn: base_predictions["ItemKNN"][slug] || 0,
                autorec: base_predictions["AutoRec"][slug] || 0,
                contentknn: base_predictions["ContentKNN"][slug] || 0,
                userknn: base_predictions["UserKNN"][slug] || 0,
                is_watched: isWatched // 传给前端打个标签，方便 UI 渲染时区分
            };
        }

        return finalResults;
    }
}
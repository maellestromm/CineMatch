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
        const feeds = { user_ratings: tensor }; // key 必须和导出时的 input_names 一致
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
            allScores[this.movieSlugs[i]] =predictions[i];
        }

        return allScores;
    }
}

export class LightGBM {
    constructor() {
        this.auto_rec = new AutoRec();
    }

    async initialize() {
        // 等待底层所有基座模型就绪
        await this.auto_rec.initialize();
        // TODO: 未来在这里加载 LightGBM 决策树的 .txt 规则
    }

    async get_recommendations(user_profile) {
        // 拿到 AutoRec 给全量 3334 部电影打的分数
        const autoRecScores = await this.auto_rec.get_recommendations(user_profile);

        // TODO: 等 LightGBM 逻辑写好后，这里会遍历 3334 部电影，把 autoRecScores[slug] 当作特征喂给树模型

        // 暂时直接透传 AutoRec 的字典，跑通前端链路
        return autoRecScores;
    }
}
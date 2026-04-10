import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.mjs';

export class AutoRec {
    constructor() {
        this.session = null;
        this.movieSlugs = [];
        this.movieToIdx = {};
        this.numMovies = 0;
    }

    async initialize(
        modelUrl = './autorec.onnx',
        dictUrl = './movie_dictionary.json'
    ) {
        console.log("[AutoRec] Initializing");
        try {
            const dictResponse = await fetch(dictUrl);
            this.movieSlugs = await dictResponse.json();
            this.numMovies = this.movieSlugs.length;


            this.movieSlugs.forEach((slug, idx) => {
                this.movieToIdx[slug] = idx;
            });


            this.session = await ort.InferenceSession.create(modelUrl);
            console.log(`[AutoRec] Initialized`);
        } catch (error) {
            console.error("[AutoRec] Initialization failed:", error);
            throw error;
        }
    }

    async get_recommendations(user_profile) {
        if (!this.session) {
            console.warn("[AutoRec] Un-Initialize");
            return {};
        }


        const targetVector = new Float32Array(this.numMovies);
        const watchedIndices = [];
        let hasInput = false;


        for (const [slug, rating] of Object.entries(user_profile)) {
            if (this.movieToIdx[slug] !== undefined) {
                const idx = this.movieToIdx[slug];
                targetVector[idx] = parseFloat(rating);
                watchedIndices.push(idx);
                hasInput = true;
            }
        }

        if (!hasInput) {
            console.warn("[AutoRec] Movie not in Model");
            return {};
        }


        const tensor = new ort.Tensor('float32', targetVector, [1, this.numMovies]);
        const feeds = {user_ratings: tensor};
        const results = await this.session.run(feeds);


        const predictions = results.predictions.data;


        for (const idx of watchedIndices) {
            predictions[idx] = -999.0;
        }


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
        this.simMatrix = {};
    }

    /**
     * 初始化加载
     */
    async initialize(
        dictUrl = './movie_dictionary.json',
        simUrl = './content_knn_k1.json'
    ) {
        console.log("[ContentKNN-Hit] Initializing");
        try {
            const dictResponse = await fetch(dictUrl);
            this.movieSlugs = await dictResponse.json();
            this.numMovies = this.movieSlugs.length;

            this.movieSlugs.forEach((slug, idx) => {
                this.movieToIdx[slug] = idx;
            });

            const simResponse = await fetch(simUrl);
            this.simMatrix = await simResponse.json();

            console.log(`[ContentKNN-Hit] Initialized`);
        } catch (error) {
            console.error("[ContentKNN-Hit] Initialization failed:", error);
            throw error;
        }
    }


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


        for (let i = 0; i < this.numMovies; i++) {

            if (watchedIndices.includes(i)) {
                allScores[this.movieSlugs[i]] = -999.0;
                continue;
            }

            const neighborData = this.simMatrix[i];
            let finalScore = 0.0;

            if (neighborData) {

                const [bestJ, sim] = neighborData;


                if (hasRatedMask[bestJ] > 0) {
                    const recScore = sim * targetVector[bestJ];
                    if (sim >= 0.01) {
                        finalScore = (recScore + damping * prior_mean) / (sim + damping);
                    } else {
                        finalScore = prior_mean;
                    }
                } else {

                    finalScore = prior_mean;
                }
            } else {

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
        this.simMatrix = {};
    }


    async initialize(
        dictUrl = './movie_dictionary.json',
        simUrl = './item_knn_k7.json'
    ) {
        console.log("[ItemKNN-Hit] Initializing");
        try {
            const dictResponse = await fetch(dictUrl);
            this.movieSlugs = await dictResponse.json();
            this.numMovies = this.movieSlugs.length;

            this.movieSlugs.forEach((slug, idx) => {
                this.movieToIdx[slug] = idx;
            });

            const simResponse = await fetch(simUrl);
            this.simMatrix = await simResponse.json();

            console.log(`[ItemKNN-Hit] Initialized`);
        } catch (error) {
            console.error("[ItemKNN-Hit] Initialization failed:", error);
            throw error;
        }
    }

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

        for (let i = 0; i < this.numMovies; i++) {
            if (watchedIndices.includes(i)) {
                allScores[this.movieSlugs[i]] = -999.0;
                continue;
            }

            const neighbors = this.simMatrix[i];
            let finalScore = prior_mean;

            if (neighbors && neighbors.length > 0) {
                let recScore = 0.0;
                let simSum = 0.0;

                for (let k = 0; k < neighbors.length; k++) {
                    const bestJ = neighbors[k][0];
                    const sim = neighbors[k][1];

                    if (hasRatedMask[bestJ] > 0) {
                        recScore += sim * targetVector[bestJ];
                        simSum += sim;
                    }
                }


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
        this.vt = [];
        this.k_factors = 39;
    }

    /**
     * 初始化加载
     */
    async initialize(
        dictUrl = './movie_dictionary.json',
        svdUrl = './svd_k39_vt.json'
    ) {
        console.log("[SVD] Initializing");
        try {
            const dictResponse = await fetch(dictUrl);
            this.movieSlugs = await dictResponse.json();
            this.numMovies = this.movieSlugs.length;

            this.movieSlugs.forEach((slug, idx) => {
                this.movieToIdx[slug] = idx;
            });

            const svdResponse = await fetch(svdUrl);
            this.vt = await svdResponse.json();


            this.k_factors = this.vt.length;

            console.log(`[SVD] Initialized`);
        } catch (error) {
            console.error("[SVD] Initialization failed:", error);
            throw error;
        }
    }

    async get_recommendations(user_profile) {
        if (!this.numMovies || !this.vt.length) return {};

        const targetVector = new Float32Array(this.numMovies);
        const watchedIndices = [];

        let ratingsSum = 0;
        let userRatingsCount = 0;


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


        const prior_mean = userRatingsCount > 0 ? (ratingsSum / userRatingsCount) : 3.0;
        const targetCentered = new Float32Array(this.numMovies);

        for (let i = 0; i < this.numMovies; i++) {
            if (targetVector[i] > 0) {
                targetCentered[i] = targetVector[i] - prior_mean;
            } else {
                targetCentered[i] = 0.0;
            }
        }


        const hiddenVector = new Float32Array(this.k_factors);
        for (let k = 0; k < this.k_factors; k++) {
            let sum = 0.0;
            const vtRow = this.vt[k];
            for (let i = 0; i < this.numMovies; i++) {
                sum += targetCentered[i] * vtRow[i];
            }
            hiddenVector[k] = sum;
        }


        const reconstructedScores = new Float32Array(this.numMovies);
        for (let i = 0; i < this.numMovies; i++) {
            let sum = 0.0;
            for (let k = 0; k < this.k_factors; k++) {
                sum += hiddenVector[k] * this.vt[k][i];
            }
            reconstructedScores[i] = sum;
        }


        const allScores = {};
        for (let i = 0; i < this.numMovies; i++) {
            if (watchedIndices.includes(i)) {
                allScores[this.movieSlugs[i]] = -999.0;
            } else {

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
        this.users = [];
        this.k_neighbors = 13;
    }

    async initialize(
        dictUrl = './movie_dictionary.json',
        usersUrl = './user_knn_k13.json'
    ) {
        console.log("[UserKNN-Hit] Initializing");
        try {
            const dictResponse = await fetch(dictUrl);
            this.movieSlugs = await dictResponse.json();
            this.numMovies = this.movieSlugs.length;

            this.movieSlugs.forEach((slug, idx) => {
                this.movieToIdx[slug] = idx;
            });

            const usersResponse = await fetch(usersUrl);
            this.users = await usersResponse.json();

            console.log(`[UserKNN-Hit] Initialized`);
        } catch (error) {
            console.error("[UserKNN-Hit] Initialization failed:", error);
            throw error;
        }
    }

    async get_recommendations(user_profile) {
        if (!this.numMovies || this.users.length === 0) return {};

        const targetVector = new Float32Array(this.numMovies);
        const watchedIndices = [];
        let targetNormSq = 0;
        let ratingsSum = 0;
        let userRatingsCount = 0;

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

        const similarities = new Float32Array(this.users.length);
        for (let i = 0; i < this.users.length; i++) {
            const user = this.users[i];
            let dotProduct = 0.0;

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

        const indices = Array.from({length: this.users.length}, (_, i) => i);
        indices.sort((a, b) => similarities[b] - similarities[a]);
        const topKIndices = indices.slice(0, this.k_neighbors);

        const recommendationScores = new Float32Array(this.numMovies);
        const similaritySums = new Float32Array(this.numMovies);

        for (let k = 0; k < topKIndices.length; k++) {
            const userIdx = topKIndices[k];
            const sim = similarities[userIdx];

            if (sim <= 0) continue;

            const user = this.users[userIdx];

            for (const jStr of Object.keys(user.ratings)) {
                const j = parseInt(jStr);
                const rating = user.ratings[j];

                recommendationScores[j] += sim * rating;
                similaritySums[j] += sim;
            }
        }

        const damping = 3.0;
        const allScores = {};

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


export class NN_Meta {
    constructor() {
        this.session = null;
        this.movieSlugs = [];
        this.numMovies = 0;
        this.STD_CLIP_LOWER = 0.1;

        this.svd = new SVDRecommender();
        this.itemKnn = new ItemKNN_Hit();
        this.autoRec = new AutoRec();
        this.contentKnn = new ContentKNN_Hit();
        this.userKnn = new UserKNN_Hit();
    }

    async initialize() {
        console.log("[NN-Meta] Initializing");
        try {
            const dictUrl = './movie_dictionary.json';

            await Promise.all([
                this.svd.initialize(dictUrl, './svd_k39_vt.json'),
                this.itemKnn.initialize(dictUrl, './item_knn_k7.json'),
                this.autoRec.initialize('./autorec.onnx', dictUrl),
                this.contentKnn.initialize(dictUrl, './content_knn_k1.json'),
                this.userKnn.initialize(dictUrl, './user_knn_k13.json'),

                (async () => {
                    const dictResponse = await fetch(dictUrl);
                    this.movieSlugs = await dictResponse.json();
                    this.numMovies = this.movieSlugs.length;
                    this.session = await ort.InferenceSession.create('./nn_meta.onnx');
                })()
            ]);

            console.log(`[NN-Meta] Initialized`);
        } catch (error) {
            console.error("[NN-Meta] Initialization failed:", error);
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

    async get_recommendations(user_profile) {
        if (!this.session) {
            console.warn("[NN-Meta] Un-Initialize");
            return [];
        }

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

        for (const modelName of modelNames) {
            const preds = base_predictions[modelName];
            if (!preds) continue;

            const scores = new Float32Array(this.numMovies);

            for (let i = 0; i < this.numMovies; i++) {
                const slug = this.movieSlugs[i];
                let s = preds[slug];

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

        const tensor = new ort.Tensor('float32', batchedInput, [this.numMovies, 5]);
        const results = await this.session.run({features: tensor});
        const nnScores = results.score.data;

        const finalResults = {};

        for (let i = 0; i < this.numMovies; i++) {
            const slug = this.movieSlugs[i];

            let absoluteScore;
            let isWatched = false;

            if (user_profile[slug] !== undefined) {
                absoluteScore = parseFloat(user_profile[slug]);
                isWatched = true;
            } else {
                absoluteScore = (nnScores[i] * user_std) + user_avg;
            }

            finalResults[slug] = {
                meta: absoluteScore,
                svd: base_predictions["SVD"][slug] || 0,
                itemknn: base_predictions["ItemKNN"][slug] || 0,
                autorec: base_predictions["AutoRec"][slug] || 0,
                contentknn: base_predictions["ContentKNN"][slug] || 0,
                userknn: base_predictions["UserKNN"][slug] || 0,
                is_watched: isWatched
            };
        }

        return finalResults;
    }
}
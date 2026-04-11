# CineMatch

Live demo at https://donywang922.github.io/CineMatch/

## Requirements

pandas~=3.0.1\
scikit-learn~=1.8.0\
letterboxdpy~=6.4.1\
torch~=2.10.0\
matplotlib~=3.10.8\
numpy~=2.4.2\
scipy~=1.17.1.0\
lightgbm~=4.6.0\
m2cgen~=0.10.0\
onnx~=1.21.0\
seaborn~=0.13.2

## Project Structure

├── db_backup/ # Database backup files (raw crawled data)\
├── data/ # Runtime data, model weights, generated dictionaries, etc.\
├── doc/ # Charts and screenshots \
├── gui/ # User interaction through terminal I/O\
├── webui/ # User interaction through web browser\
├── models/\
│ ├── content_knn/ # Content-based recommendation (Content-KNN)\
│ ├── item_knn/ # Item-based collaborative filtering (Item-KNN)\
│ ├── svd/ # Latent factor model / Matrix factorization (Truncated SVD)\
│ ├── user_knn/ # User-based collaborative filtering (User-KNN)\
│ ├── auto_rec/ # Deep learning autoencoder (Deep AutoRec)\
│ ├── lgbm_meta/ # LightGBM Meta learner\
│ └── nn_meta/ # Deep&Wide Meta learner\
├── tools/\
│ ├── clear_db.py # Database cleaning and preprocessing script\
│ ├── split_db.py # Train/test set physical split script\
│ ├── Movie_first_crawler.py # Movie-first crawler\
│ └── User_first_crawler.py # User-first crawler\
├── util.py # Global helper and path configuration functions\
└── README.md # Project documentation

## Recommendation Models

This project implements five classic recommendation algorithms spanning different eras to build a complete recall and
ranking architecture:

### 1. Content-KNN

* **Principle**: "These movies share the same tags as the ones you've watched before."
* **Implementation**: Extracts movie metadata such as director, genre, cast, and overview to build TF-IDF text feature
  vectors. By calculating cosine similarity, it finds candidate movies that are physically closest to the user's
  historical high-rated movies. This solves the cold-start problem for new movies.

### 2. Item-KNN

* **Principle**: "People who like this movie usually also like that movie."
* **Implementation**: Calculates the cosine similarity between items based on the global Item-User interaction matrix.
  Introduces Bayesian Smoothing to penalize high-score biases from small samples. Achieves ultra-fast recommendations
  during inference via tensor matrix multiplication.

### 3. User-KNN

* **Principle**: "What are your soulmates with similar tastes watching?"
* **Implementation**: Finds the K nearest neighbors with the closest rating trends to the target user based on the
  User-Item interaction matrix. Also incorporates a prior mean and damping factor to eliminate popularity bias. Supports
  dual-engine backends: CPU (scikit-learn) and GPU (PyTorch tensor operations).

### 4. Matrix Factorization (SVD)

* **Principle**: "You might not understand your own taste, but math does."
* **Implementation**: Uses Truncated Singular Value Decomposition (Truncated SVD) to reduce the dimensionality of the
  massive rating matrix, extracting 39 latent semantic dimensions (Latent Factors). Utilizes the Folding-in projection
  technique during the inference phase to achieve ultra-low latency score predictions for new users without retraining.

### 5. Deep Autoencoder (Deep AutoRec)

* **Principle**: "Using neural networks to learn the non-linear compression and reconstruction of high-dimensional
  features."
* **Implementation**: Constructs a multi-layer Encoder-Decoder architecture and introduces 30% Dropout to prevent
  overfitting. It can capture complex, non-linear implicit correlations in user rating data with extreme precision.

### 6. Meta Learner (LightGBM)

* **Principle**: "Minimizing prediction error through ensemble learning, even if it means sacrificing ranking contrast."
* **Implementation**: Extracts the prediction scores from the 5 base models as input features and trains a Gradient
  Boosting Decision Tree (GBDT). Because it is strictly optimized for Mean Squared Error (MSE), the model learns to play
  it safe, heavily penalizing extreme scores. As a result, it achieves the most accurate absolute score predictions
  across the entire database. However, this "conservative" nature flattens the prediction variance, making
  it struggle to push hidden masterpieces to the top, resulting in a mediocre Hit Rate.

### 7. Deep & Wide (NN Meta)

* **Principle**: "Using non-linear transformations and dynamic variance scaling to find the perfect boundary between
  masterpieces and bad movies."
* **Implementation**: Employs a Wide & Deep neural network architecture to act as the ultimate judge. Before feeding the
  base model predictions into the network, it dynamically applies Z-score normalization to amplify the score
  variance—effectively resolving the zero-sum conflict between RMSE and Hit Rate. The network captures complex,
  non-linear feature interactions to determine the final trust weights for each base model. It dominates the leaderboard
  with a massive 78.96% Hit Rate.

## Crawlers

To collect high-quality real rating data, we designed two complementary crawling strategies based on alternating queue
fetching:

* **User-First Crawler**
    1. Fetches all movies reviewed by all users in the User Queue.
    2. Selects the movie with the most reviews, retrieves its detailed metadata, extracts popular reviews under that
       movie, and adds the authors of these reviews to the User Queue.

    * **Purpose**: Rapidly aggregates user groups that have interacted with core movies, quickly increasing the local
      density of the User-Item matrix.

* **Movie-First Crawler**
    1. Retrieves all movies reviewed by a specific user in the User Queue and adds these movies to the Movie Queue.
    2. Fetches detailed metadata for all movies in the Movie Queue, extracts popular reviews for these movies, and adds
       the review authors to the User Queue.

    * **Purpose**: Focuses on exploring movie diversity, broadly expanding the boundaries of movie genres in the
      database, and providing rich feature materials for Content-KNN.

## Before Running
1. Install the required packages listed in `requirements.txt`.
2. Install the appropriate PyTorch installation following the instructions at https://pytorch.org/get-started/locally/.
3. Extract `db_backup/user_first_cut3_clear.7z` into the `data/` directory.
4. Run `tools/split_db.py`. This performs a strict physical split of the database at a 9:1 ratio, generating
   `models/train_model.db` and `models/test_eval.db` to ensure zero data leakage during evaluation.

## How to Run Models
1. All models come with a main function and can be run directly.
2. For auto_rec, run `models/auto_rec/infer_autorec.py`
3. For svd, run `models/svd/recommender.py`
4. For content_knn, run `models/content_knn/recommender.py`
5. For item_knn, run `models/item_knn/recommender.py`
6. For user_knn, run `models/user_knn/recommender.py`
7. For lgbm_meta, run `models/lgbm_meta/lgbm_recommender_rmse.py`
8. For nn_meta, run `models/nn_meta/nn_recommender.py`

## How to Run Benchmarks
1. Run `models/evaluate_strict.py` to view the leaderboard of all models on Hit Rate and Precision.
2. Run `models/evaluate_rmse.py` to view the leaderboard of all models on the true taste prediction accuracy (1-5 stars).

## How to Run Terminal Interface
1. Extract `db_backup/user_first_cut3_clear.7z` into the `data/` directory.
2. Run `tools/split_db.py`. This performs a strict physical split of the database at a 9:1 ratio, generating
   `train_model.db` and `test_eval.db` to ensure zero data leakage during evaluation.
3. Run `gui/user_io.py`
4. Enter Letterboxd username.

## How to Run Web Interface
1. Run `python -m http.server 8000 -d webui`
2. Access http://localhost:8000

## How to Train Meta Learner
1. The meta learner requires meta_dataset.db for training. 
2. You can directly extract `db_backup/meta_dataset.7z` to the `data/` folder.
3. Alternatively, you can run `prepare_meta_data.py` to generate a new one.
4. Run `train_meta_learner_regression.py` to train LightGBM meta-learner.
5. Run `train_meta_learner_nn.py` to train NN meta-learner.

## Performance

### 1. Recall & Hit Rate Test (Hit Rate & Precision @ 10)

Objective: Can the model blindly guess the true interacted movies hidden in the test set out of a vast sea of movies?

### 2. True Taste Prediction Accuracy (RMSE Score)

Objective: Given that a user has watched a movie, can the model accurately predict their specific 1-5 star rating? (
Lower score is better)

| Model Name      | Hit Rate (@10) | Precision (@10) | Avg Latency | RMSE Score |
|:----------------|:---------------|:----------------|:------------|:-----------|
| NN-Meta         | **78.96%**     | 23.67%          | 632.9 ms    | 0.7537     |
| LightGBM-rmse   | 67.54%         | 17.88%          | 987.6 ms    | **0.6997** |
| SVD             | 76.51%         | 25.64%          | 16.8 ms     | 0.8607     |
| Deep-AutoRec    | 48.29%         | 8.71%           | 4.0 ms      | 0.7598     |
| User-KNN-13     | 75.53%         | 18.19%          | 17.2 ms     | 0.8228     |
| User-KNN-168    | 62.81%         | 13.13%          | 13.7 ms     | 0.7701     |
| Item-KNN-7      | 73.41%         | 19.43%          | 18.9 ms     | 0.9180     |
| Item-KNN-50     | 55.46%         | 11.68%          | 30.3 ms     | 0.8928     |
| Content-KNN-1   | 44.70%         | 7.50%           | 1.1 ms      | 1.0165     |
| Content-KNN-871 | 19.09%         | 2.69%           | 0.9 ms      | 0.9551     |
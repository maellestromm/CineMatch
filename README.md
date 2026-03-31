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

├── db_backup/ # Database backup files (raw crawled data)\
├── data/ # Runtime data, model weights, generated dictionaries, etc.\
├── models/\
│ ├── content_knn/ # Content-based recommendation (Content-KNN)\
│ ├── item_knn/ # Item-based collaborative filtering (Item-KNN)\
│ ├── svd/ # Latent factor model / Matrix factorization (Truncated SVD)\
│ ├── user_knn/ # User-based collaborative filtering (User-KNN)\
│ ├── auto_rec/ # Deep learning autoencoder (Deep AutoRec)\
│ ├── meta/ # Weighted hybrid ensemble recommender system\
│ └── saved_models/ Stored .pkl files for each model
├── tools/\
│ ├── clear_db.py # Database cleaning and preprocessing script\
│ ├── split_db.py # Train/test set physical split script\
│ ├── Movie_first_crawler.py # Movie-first crawler\
│ └── User_first_crawler.py # User-first crawler\
├── gui/ User interaction through terminal and GUI I/O\
├── visualizations/ Snapshots and graphs showcasing model performance\
├── util.py # Global helper and path configuration functions\
└── README.md # Project documentation

## How to Run

Setup:
1. Ensure you have 7z installed
2. (Optional) Run "make setup" to install dependencies
3. Run "make initialize" to extract/split the data and extract the MetaRecommender model

Run in Terminal:
1. Run "make run" to produce user-based recommendations with MetaRecommender

Evaluation:
1. Run "make eval" to run both Hit Rate@10/Precision@10 and RMSE benchmarks

<img alt="Terminal Demo" src="https://github.com/maellestromm/CineMatch/blob/65867eb2e5f0253ed5f50fd21d6a7a4326193d27/visualizations/demo-terminal-io.png" width="30%"></img>

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

### 4. Matrix Factorization (SVD-50)

* **Principle**: "You might not understand your own taste, but math does."
* **Implementation**: Uses Truncated Singular Value Decomposition (Truncated SVD) to reduce the dimensionality of the
  massive rating matrix, extracting 50 latent semantic dimensions (Latent Factors). Utilizes the Folding-in projection
  technique during the inference phase to achieve ultra-low latency score predictions for new users without retraining.

### 5. Deep Autoencoder (Deep AutoRec)

* **Principle**: "Using neural networks to learn the non-linear compression and reconstruction of high-dimensional
  features."
* **Implementation**: Constructs a multi-layer Encoder-Decoder architecture and introduces 30% Dropout to prevent
  overfitting. It can capture complex, non-linear implicit correlations in user rating data with extreme precision.

## Crawlers

To obtain high-quality real rating data, we designed two complementary crawling strategies based on alternating queue
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

## Performance

### 1. Recall & Hit Rate Test (Hit Rate & Precision @ 10)

Objective: Can the model blindly guess the true interacted movies hidden in the test set out of a vast sea of movies?

| Model Name   | Hit Rate (@10) | Precision (@10) | Avg Latency |
|:-------------|:---------------|:----------------|:------------|
| **Meta**     | **78.29%**     | **27.53%**      | **863.4 ms**|
| SVD-50       | 77.47%         | 26.38%          | 9.2 ms      |
| User-KNN     | 72.20%         | 19.01%          | 60.8 ms     |
| Item-KNN     | 69.90%         | 18.63%          | 46.2 ms     |
| Deep AutoRec | 46.38%         | 9.38%           | 2.9 ms      |
| Content-KNN  | 22.70%         | 3.21%           | 7.3 ms      |

### 2. True Taste Prediction Accuracy (RMSE Score)

Objective: Given that a user has watched a movie, can the model accurately predict their specific 1-5 star rating? (
Lower score is better)

| Model Name       | RMSE Score |
|:-----------------|:-----------|
| **Deep AutoRec** | **0.7710** |
| User-KNN         | 0.8251     |
| SVD-50           | 0.8604     |
| Item-KNN         | 0.9175     |
| Meta             | 0.9260     |
| Content-KNN      | 0.9633     |

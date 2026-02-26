# CineMatch

## requirement

letterboxdpy 6.4.1\
scikit-learn 1.8.0\
pandas 3.0.1

## Project Structure

├── data # Database backup\
├── src/Movie_first_crawler.py # Movie first spider\
├── src/User_first_crawler.py # User first spider\
├── src/content_knn.py # Content Based KNN demo\
├── src/user_knn.py # User Based KNN demo\
├── src/clear_db.py # Clean up movies and their reviews in the database that lack details\
├── src/split_db.py # Split users into two databases in an 8:2 ratio\
├── src/evaluate_strict.py # Accuracy test

## About spider

movie first: 1. Fetch all movies reviewed by a user from the user queue to the movie queue. 2. Fetch details of all
movies from the movie queue and add the top reviews to the user queue.

user first: 1. Retrieve all movies commented on by all users in the user queue. 2. Select the movie with the most
comments and retrieve its details and popular comments to the user queue.

## How to run evaluate_strict.py

1. run split_db.py\
   split_db.py takes a db file and splits the user ratings in it
   into `train_model.db` and `test_eval.db` in an 8:2 ratio.
2. run evaluate_strict.py
   `evaluate_strict.py` uses `train_model.db` to build a k-nearest neighbor (KNN) and then uses user ratings from
   `test_eval.db` to test the accuracy of the KNN. For each user, 80% of the ratings are randomly selected and provided
   to the model, and then the model's predictions are checked for overlap with the remaining 20% of the ratings. You
   can change the model to be tested on line 20.
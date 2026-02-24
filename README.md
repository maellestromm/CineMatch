# CineMatch

## requirement

letterboxdpy 6.4.1\
scikit-learn 1.8.0\
pandas 3.0.1

## Project Structure

├── data # Database backup\
├── src/Movie_first_crawler.py # Movie first spider\
├── src/User_first_crawler.py # User first spider\
├── src/content_knn.py # Content Based KNN demo

## About spider

movie first: 1. Fetch all movies reviewed by a user from the user queue to the movie queue. 2. Fetch details of all
movies from the movie queue and add the top reviews to the user queue.

user first: 1. Retrieve all movies commented on by all users in the user queue. 2. Select the movie with the most
comments and retrieve its details and popular comments to the user queue.
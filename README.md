# Movie Recommendation System with Collaborative Filtering

This repository contains a movie recommendation system implemented using **collaborative filtering** as part of the **Machine Learning Specialization by Deeplearning.ai and Stanford University**. The system predicts user preferences for movies based on their past interactions and uses a dataset derived from the MovieLens dataset.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Acknowledgements](#acknowledgements)

---

## Overview
This project builds a collaborative filtering recommendation system that:
1. Learns user preferences and movie features based on existing ratings.
2. Predicts how users might rate movies they haven't watched.
3. Generates personalized movie recommendations.

The collaborative filtering approach involves:
- Representing movies as feature vectors.
- Representing users with parameter vectors and biases.
- Using a cost function with regularization to train the model and minimize prediction errors.

---

## Features
- **Cost Function**: Implements collaborative filtering with support for vectorized operations.
- **Custom Training Loop**: Uses TensorFlow for gradient calculations and optimization.
- **Recommendation System**: Predicts user ratings for movies and provides top recommendations.
- **Custom Ratings**: Allows users to add their ratings to personalize recommendations.

---

## Dataset
The dataset used in this project is a reduced version of the [MovieLens](https://grouplens.org/datasets/movielens/latest/) dataset:
- **Movies**: 4778 movies (released after the year 2000).
- **Users**: 443 users.
- **Ratings**: Scale from 0.5 to 5 in 0.5 increments.
- Matrices:
  - `Y`: Ratings matrix.
  - `R`: Binary indicator matrix (1 if a rating exists).

Pre-computed values of movie and user parameters (`X`, `W`, `b`) are also included for faster experimentation.

---

## Usage
1. Run the main script:
   ```bash
   python collaborative_filtering.py
   ```

2. Add custom ratings in the `my_ratings` section of the script to personalize recommendations.

3. View top movie recommendations in the output.

---

## Project Structure
```plaintext
movie-recommendation-collaborative-filtering/
│
├── collaborative_filtering.py      # Main script implementing the system
├── recsys_utils.py                 # Helper functions for data processing and loading
├── data/                           # Directory for dataset files
│   ├── small_movies_X.csv          # Movie features
│   ├── small_movies_W.csv          # User parameters
│   ├── small_movies_b.csv          # User biases
│   ├── small_movies_Y.csv          # Ratings matrix
│   ├── small_movies_R.csv          # Ratings indicator matrix
│   └── small_movie_list.csv        # Movie metadata
├── README.md                       # Project documentation
└── requirements.txt                # Python dependencies
```

---

## Acknowledgements
- The results of this project are generated as part of a skills lab of the **Machine Learning Specialization** 
  offered by [Deeplearning.ai](https://www.deeplearning.ai/) and **Stanford University**.
- The dataset is derived from the [MovieLens Dataset](https://grouplens.org/datasets/movielens/latest/), maintained by GroupLens Research.
- TensorFlow is used for optimization and gradient calculation.


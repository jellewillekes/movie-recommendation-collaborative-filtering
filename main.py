# Practice lab: Collaborative Filtering Recommender Systems

"""
This script implements collaborative filtering to build a recommender system for movies.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from recsys_utils import load_precalc_params_small, load_ratings_small, normalizeRatings, load_Movie_List_pd

# Section 1 - Notation
"""
- r(i,j): 1 if user j rated movie i; 0 otherwise
- y(i,j): rating given by user j on movie i (defined if r(i,j) = 1)
- X: matrix of movie feature vectors
- W: matrix of user parameter vectors
- b: vector of user bias terms
- R: matrix indicating whether a movie has been rated by a user
"""

# Section 2 - Recommender Systems
"""
The goal of a collaborative filtering recommender system is to generate:
1. A parameter vector for each user that embodies the user's preferences.
2. A feature vector for each movie describing its attributes.
"""

# Section 3 - Movie ratings dataset
X, W, b, num_movies, num_features, num_users = load_precalc_params_small()
Y, R = load_ratings_small()

print("Dataset shapes:")
print(f"Y: {Y.shape}, R: {R.shape}")
print(f"X: {X.shape}, W: {W.shape}, b: {b.shape}")
print(f"num_features: {num_features}, num_movies: {num_movies}, num_users: {num_users}")


# Section 4 - Collaborative filtering learning algorithm
def cofi_cost_func(X, W, b, Y, R, lambda_):
    """
    Collaborative filtering cost function.

    Args:
      X: matrix of movie features
      W: matrix of user parameters
      b: vector of user biases
      Y: matrix of user ratings
      R: matrix indicating rated movies
      lambda_: regularization parameter

    Returns:
      J: cost value
    """
    nm, nu = Y.shape
    J = 0

    for i in range(nm):
        for j in range(nu):
            if R[i, j] == 1:  # Only consider ratings where R(i, j) = 1
                prediction = np.dot(W[j, :], X[i, :]) + b[0, j]
                error = prediction - Y[i, j]
                J += (error ** 2) / 2

    J += (lambda_ / 2) * (np.sum(np.square(W)) + np.sum(np.square(X)))
    return J


# Evaluate cost function
num_users_r, num_movies_r, num_features_r = 4, 5, 3
X_r = X[:num_movies_r, :num_features_r]
W_r = W[:num_users_r, :num_features_r]
b_r = b[0, :num_users_r].reshape(1, -1)
Y_r = Y[:num_movies_r, :num_users_r]
R_r = R[:num_movies_r, :num_users_r]

print("\nCost function evaluations:")
J = cofi_cost_func(X_r, W_r, b_r, Y_r, R_r, 0)
print(f"Cost (lambda=0): {J:.2f}")

J = cofi_cost_func(X_r, W_r, b_r, Y_r, R_r, 1.5)
print(f"Cost (with regularization): {J:.2f}")


# Vectorized implementation
def cofi_cost_func_v(X, W, b, Y, R, lambda_):
    """
    Vectorized collaborative filtering cost function.

    Args:
      X: matrix of movie features
      W: matrix of user parameters
      b: vector of user biases
      Y: matrix of user ratings
      R: matrix indicating rated movies
      lambda_: regularization parameter

    Returns:
      J: cost value
    """
    j = (tf.linalg.matmul(X, tf.transpose(W)) + b - Y) * R
    J = 0.5 * tf.reduce_sum(j ** 2) + (lambda_ / 2) * (tf.reduce_sum(X ** 2) + tf.reduce_sum(W ** 2))
    return J


# Section 5 - Learning movie recommendations
"""
Train the collaborative filtering model to learn parameters X, W, and b.
"""

# Adding custom ratings
movieList, movieList_df = load_Movie_List_pd()
my_ratings = np.zeros(num_movies)

# Sample ratings
my_ratings[929] = 5
my_ratings[246] = 5
my_ratings[2716] = 3

print("\nNew user ratings:")
for i, rating in enumerate(my_ratings):
    if rating > 0:
        print(f"Rated {rating} for movie ID {i}")

# Add ratings to dataset
Y = np.c_[my_ratings, Y]
R = np.c_[(my_ratings != 0).astype(int), R]

# Normalize ratings
Ynorm, Ymean = normalizeRatings(Y, R)

# Initialize parameters
tf.random.set_seed(1234)
W = tf.Variable(tf.random.normal((num_users + 1, num_features), dtype=tf.float64))
X = tf.Variable(tf.random.normal((num_movies, num_features), dtype=tf.float64))
b = tf.Variable(tf.random.normal((1, num_users + 1), dtype=tf.float64))

optimizer = keras.optimizers.Adam(learning_rate=1e-1)

# Custom training loop
iterations = 200
lambda_ = 1
for iter in range(iterations):
    with tf.GradientTape() as tape:
        cost_value = cofi_cost_func_v(X, W, b, Ynorm, R, lambda_)

    grads = tape.gradient(cost_value, [X, W, b])
    optimizer.apply_gradients(zip(grads, [X, W, b]))

    if iter % 20 == 0:
        print(f"Iteration {iter}, Cost: {cost_value:.2f}")

# Section 6 - Recommendations
"""
Compute predictions for all movies and display recommendations.
"""

p = np.matmul(X.numpy(), np.transpose(W.numpy())) + b.numpy()
pm = p + Ymean
my_predictions = pm[:, 0]
ix = tf.argsort(my_predictions, direction='DESCENDING')

print("\nTop recommendations:")
for i in range(10):
    idx = ix[i]
    print(f"Predicting rating {my_predictions[idx]:.2f} for movie {movieList[idx]}")

## Principal Component Analysis (PCA) for Movie Recommender System

### Overview

#### This repository contains code for performing Principal Component Analysis (PCA) on a movie dataset and implementing a recommender system function based on the PCA results.
Overview

#### Principal Component Analysis (PCA) is a dimensionality reduction technique commonly used in data analysis and machine learning. In this project, PCA is applied to a movie dataset to identify patterns and relationships among movies and users.

#### The steps involved in the project include:

1. Loading and preprocessing the movie dataset.
2. Performing PCA to reduce the dimensionality of the dataset.
3. Visualizing the results using scree plots, loading plots, and score plots.
4. Implementing a recommender system function based on the PCA results.

### Dataset

#### The dataset used in this project consists of the following files:

- dataMatrix.csv: Matrix of user ratings for movies.
- selectedMovies.csv: Information about selected movies.
- usertype.csv: Information about user types.

### Results

#### Upon running the main script, the following results will be displayed:

- Scree plot: Explained variance ratio of principal components.
  ![Scree plot](https://github.com/aminfdev/Principal-Component-Analysis/blob/main/reports/report_1.png)
- Loading plot: Visualization of feature loadings on principal components.
  ![Loading plot](https://github.com/aminfdev/Principal-Component-Analysis/blob/main/reports/report_2.png)
- Score plots: Visualization of projected data points onto principal components.
  ![Score plot 1](https://github.com/aminfdev/Principal-Component-Analysis/blob/main/reports/report_3.png)
  ![Score plot 2](https://github.com/aminfdev/Principal-Component-Analysis/blob/main/reports/report_4.png)
- Recommended movies: List of movies recommended for a specific user based on PCA.


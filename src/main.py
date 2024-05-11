import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_data():
    # Load data
    data_matrix = pd.read_csv("./data/dataMatrix.csv")
    selected_movies = pd.read_csv("./data/selectedMovies.csv")
    user_types = pd.read_csv("./data/usertype.csv")

    return data_matrix, selected_movies, user_types


def preprocess_data(data_matrix):
    # Drop the first column from data_matrix
    data_matrix = data_matrix.drop(data_matrix.columns[0], axis=1)

    # Compute mean values and center the data
    mean_values = data_matrix.mean(numeric_only=True)
    centered_data = data_matrix - mean_values

    return data_matrix, centered_data


def perform_pca(centered_data, t=2):
    # Calculate covariance matrix
    covariance_matrix = centered_data.cov()

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

    # Sort eigenvalues and corresponding eigenvectors
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues_sorted = eigenvalues[sorted_indices]
    eigenvectors_sorted = eigenvectors[:, sorted_indices]

    # Select the top t eigenvectors (t=2 for a 2D plot)
    transformation_matrix = eigenvectors_sorted[:, :t]

    # Project data onto the new subspace
    projected_data = centered_data.dot(transformation_matrix)

    return transformation_matrix, projected_data, eigenvalues_sorted


def visualize_scree_plot(eigenvalues):
    # Calculate the explained variance ratio
    explained_variances = eigenvalues / np.sum(eigenvalues)

    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(1, len(eigenvalues) + 1),
             explained_variances, marker='o', linestyle='-')
    plt.title('Scree Plot')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.xticks(np.arange(1, len(eigenvalues) + 1))
    plt.grid(True)
    plt.show()


def visualize_loading_plot(loading_plot_data):
    plt.figure(figsize=(16, 10))
    plt.scatter(loading_plot_data['PC1'], loading_plot_data['PC2'])
    plt.title('Loading Plot')
    plt.xlabel('Principal Component 1 (PC1)')
    plt.ylabel('Principal Component 2 (PC2)')

    for feature, (pc1, pc2) in loading_plot_data.iterrows():
        plt.annotate(feature, (pc1, pc2), textcoords="offset points",
                     xytext=(5, 5), ha='center')

    plt.show()


def analyze_loading_plot(loading_plot_data, selected_movies):
    # Sort loading plot by PC1 values
    loading_plot_sorted_pc1 = loading_plot_data.sort_values(
        by='PC1', ascending=False)

    # Extract top 10 movies with the most values in PC1 (indices)
    top_pc1_movies = loading_plot_sorted_pc1.index[:10].tolist()

    # Extract top 10 movies with the least values in PC1 (indices)
    bottom_pc1_movies = loading_plot_sorted_pc1.index[-10:].tolist()
    bottom_pc1_movies = reversed(bottom_pc1_movies)

    # Sort loading plot by PC2 values
    loading_plot_sorted_pc2 = loading_plot_data.sort_values(
        by='PC2', ascending=False)

    # Extract top 10 movies with the most values in PC2 (indices)
    top_pc2_movies = loading_plot_sorted_pc2.index[:10].tolist()

    # Extract top 10 movies with the least values in PC2 (indices)
    bottom_pc2_movies = loading_plot_sorted_pc2.index[-10:].tolist()
    bottom_pc2_movies = reversed(bottom_pc2_movies)

    # Extract movie names
    movie_names = selected_movies.iloc[:, 1].tolist()

    # Modify lists with corresponding movie names
    top_pc1_movies_names = [(int(movie_index), movie_names[int(
        movie_index)]) for movie_index in top_pc1_movies]
    bottom_pc1_movies_names = [(int(movie_index), movie_names[int(
        movie_index)]) for movie_index in bottom_pc1_movies]
    top_pc2_movies_names = [(int(movie_index), movie_names[int(
        movie_index)]) for movie_index in top_pc2_movies]
    bottom_pc2_movies_names = [(int(movie_index), movie_names[int(
        movie_index)]) for movie_index in bottom_pc2_movies]

    return top_pc1_movies_names, \
        bottom_pc1_movies_names, \
        top_pc2_movies_names, \
        bottom_pc2_movies_names


def visualize_score_plot(projected_data):
    plt.figure(figsize=(16, 10))
    plt.scatter(projected_data.iloc[:, 0], projected_data.iloc[:, 1])
    plt.title('Score Plot')
    plt.xlabel('Principal Component 1 (PC1)')
    plt.ylabel('Principal Component 2 (PC2)')

    for index, (pc1, pc2) in projected_data.iterrows():
        plt.annotate(str(index), (pc1, pc2),
                     textcoords="offset points", xytext=(5, 5), ha='center')

    plt.show()


def visualize_colored_score_plot(projected_data, user_types):
    user_types.reset_index(drop=True, inplace=True)
    projected_data.reset_index(drop=True, inplace=True)

    colors = ['r', 'g', 'b']
    user_type_labels = {0: 'Teenager', 1: 'Male Adult', 2: 'Female Adult'}

    plt.figure(figsize=(16, 10))

    for user_type in range(3):
        mask = user_types['userType'] == user_type
        plt.scatter(projected_data[mask].iloc[:, 0], projected_data[mask].iloc[:,
                    1], c=colors[user_type], label=user_type_labels[user_type])

    plt.title('Score Plot with User Types')
    plt.xlabel('Principal Component 1 (PC1)')
    plt.ylabel('Principal Component 2 (PC2)')
    plt.legend()
    plt.show()


def recommend_movies(user_id, data_matrix, selected_movies, transformation_matrix, n=10):
    # Check if user_id is within range
    if user_id < 0 or user_id >= data_matrix.shape[0]:
        raise ValueError(
            f"Invalid user ID. User ID should be within the range [0, {data_matrix.shape[0]-1}]")

    # Get movie names from selected_movies DataFrame
    movie_names = selected_movies.iloc[:, 1].tolist()

    # Get user ratings
    user_ratings = data_matrix.iloc[user_id]

    # Project user ratings onto the PCA space
    user_projection = np.dot(np.transpose(transformation_matrix), user_ratings)

    # Reconstruct ratings
    reconstructed_ratings = np.dot(transformation_matrix, user_projection)

    # Find movies with high reconstructed ratings
    recommendations = []
    for i in range(len(user_ratings)):
        original_rating = user_ratings[i]
        reconstructed_rating = reconstructed_ratings[i]
        if original_rating == 0 and reconstructed_rating > 0:
            recommendations.append((i, reconstructed_rating))

    # Sort recommendations by rating (descending)
    recommendations.sort(key=lambda x: x[1], reverse=True)

    # Select top n recommendations
    top_n_recommendations = recommendations[:n]

    # Get recommended movies with their names
    recommended_movies = [(movie_index, movie_names[movie_index])
                          for movie_index, _ in top_n_recommendations]

    return recommended_movies


def main():
    # Load data
    data_matrix, selected_movies, user_types = load_data()

    # Preprocess data
    data_matrix, centered_data = preprocess_data(data_matrix)

    # Perform PCA
    transformation_matrix, projected_data, eigenvalues = perform_pca(
        centered_data)

    # Visualize Scree Plot
    visualize_scree_plot(eigenvalues[:10])

    # Visualize Loading Plot
    loading_plot_data = pd.DataFrame(transformation_matrix, columns=[
                                     'PC1', 'PC2'], index=centered_data.columns)
    visualize_loading_plot(loading_plot_data)

    # Analyze Loading Plot
    top_pc1_movies, \
        bottom_pc1_movies, \
        top_pc2_movies, \
        bottom_pc2_movies = analyze_loading_plot(
            loading_plot_data, selected_movies)

    print("Top 10 movies with the most values in PC1:")
    for movie_name in top_pc1_movies:
        print(f"{movie_name[0]:<4}: {movie_name[1]}")

    print("\nTop 10 movies with the least values in PC1:")
    for movie_name in bottom_pc1_movies:
        print(f"{movie_name[0]:<4}: {movie_name[1]}")

    print("\nTop 10 movies with the most values in PC2:")
    for movie_name in top_pc2_movies:
        print(f"{movie_name[0]:<4}: {movie_name[1]}")

    print("\nTop 10 movies with the least values in PC2:")
    for movie_name in bottom_pc2_movies:
        print(f"{movie_name[0]:<4}: {movie_name[1]}")

    # Visualize Score Plots
    visualize_score_plot(projected_data)
    visualize_colored_score_plot(projected_data, user_types)

    # Recommend Movies
    user_id = 35
    recommendations = recommend_movies(
        user_id, data_matrix, selected_movies, transformation_matrix, 10)

    print(f"\nRecommended Movies for user with ID {user_id}:")
    for movie_index, movie_name in recommendations:
        print(f"{movie_index:<4}: {movie_name}")


if __name__ == "__main__":
    main()

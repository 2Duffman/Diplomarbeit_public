import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_avg_rating_distribution(csv_path):
    """
    Reads a CSV file and plots the distribution of the 'avg_rating' column.

    Parameters:
    - csv_path (str): The file path of the CSV dataset.

    Returns:
    - A histogram displaying the distribution of 'avg_rating'.
    """
    # Load the dataset
    df = pd.read_csv(csv_path)

    # Check if 'avg_rating' exists in the dataset
    if 'avg_rating' not in df.columns:
        print("Error: 'avg_rating' column not found in the dataset.")
        return
    mean_rating = df['avg_rating'].mean()
    # Plot the distribution of 'avg_rating'
    plt.figure(figsize=(8, 5))
    sns.histplot(df['avg_rating'], edgecolor='black', alpha=0.7)
    plt.axvline(mean_rating,color='red', linestyle='--', linewidth=2, label=f"Mean = {mean_rating:.2f}")
    #plt.yscale('log')
    plt.xlabel('Avg Rating')
    plt.ylabel('Frequency')
    plt.title('Distribution of average rating')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

# Example Usage:
plot_avg_rating_distribution("./code/Machine Learning/data.csv")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def load_and_preprocess_data(file_path='data/mushrooms.csv'):
    # Read the CSV file
    dataset = pd.read_csv(file_path)

    # Explanatory Variable: Use cap-shape, cap-surface, cap-color
    X = dataset[['cap-shape', 'cap-surface', 'cap-color']]

    # Target Variable: Use 'class' as the target variable
    Y = dataset['class']

    # Create dummy variables
    X_dummies = pd.get_dummies(X)

    # Create columns for "cap-shape or not cap-shape, cap-surface or not cap-surface, cap-color or not cap-color"
    X_dummies['cap-shape or not'] = X['cap-shape'].apply(lambda x: 1 if x else 0)
    X_dummies['cap-surface or not'] = X['cap-surface'].apply(lambda x: 1 if x else 0)
    X_dummies['cap-color or not'] = X['cap-color'].apply(lambda x: 1 if x else 0)

    # Convert boolean columns to integers
    X_dummies = X_dummies.astype(int)

    # Convert Y to binary (0 or 1) as it is a binary classification model
    Y_binary = Y.apply(lambda x: 1 if x == 'p' else 0)

    # Split the data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X_dummies, Y_binary, test_size=0.3, random_state=42)

    return X_train, X_test, Y_train, Y_test
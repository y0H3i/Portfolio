# Mushroom Classification Project

## Description

This project implements a binary classification model using supervised learning to determine whether a mushroom is poisonous or edible. It uses TensorFlow to build and train a deep learning model based on specific features of mushrooms, including cap shape, cap surface, and cap color.

## Installation

1. Clone this repository:
```
git clone https://github.com/yourusername/mushroom-classification.git
```
```
cd mushroom-classification
```

2. Set up a virtual environment:
```
python -m venv venv
```

3. Activate the virtual environment:
   - On Windows:
     ```
     venv\Scripts\activate
     ```
   - On macOS and Linux:
     ```
     source venv/bin/activate
     ```

4. Install the required packages:
```
pip install -r requirements.txt
```


5. Download the dataset:
   - Download the `mushrooms.csv` file from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/mushroom)
   - Place the file in the `data/` directory of the project

## Usage

Run the main script to train and evaluate the model:
```
python main.py
```

This will:
1. Load and preprocess the data
2. Create and compile the model
3. Train the model
4. Evaluate the model on the test set

## Features

- Data preprocessing: Converts categorical variables to dummy variables
- Model architecture: Uses a deep neural network with multiple dense layers
- Early stopping: Implements early stopping to prevent overfitting
- Evaluation: Provides accuracy metrics for model performance

## Examples

Here's a code snippet showing the model architecture:

```python
def create_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(input_shape,)),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
   
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
   
    return model
```

## Example output:
```
Epoch 1/100
172/172 [==============================] - 1s 3ms/step - loss: 0.6931 - accuracy: 0.5000 - val_loss: 0.6931 - val_accuracy: 0.5000
Epoch 2/100
172/172 [==============================] - 0s 2ms/step - loss: 0.6931 - accuracy: 0.5000 - val_loss: 0.6931 - val_accuracy: 0.5000
...
Epoch 100/100
172/172 [==============================] - 0s 2ms/step - loss: 0.0987 - accuracy: 0.9724 - val_loss: 0.1104 - val_accuracy: 0.9688

Test accuracy: 0.9687999844551086
```

## License
This project is licensed under the MIT License - see the LICENSE.md file for details.


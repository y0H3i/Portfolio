from src.data_preprocessing import load_and_preprocess_data
from src.model import create_model
from src.train import train_and_evaluate_model

def main():
    X_train, X_test, Y_train, Y_test = load_and_preprocess_data()
    model = create_model(input_shape=X_train.shape[1])
    train_and_evaluate_model(model, X_train, Y_train, X_test, Y_test)

if __name__ == "__main__":
    main()
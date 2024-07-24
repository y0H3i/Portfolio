import tensorflow as tf


def train_and_evaluate_model(model, X_train, Y_train, X_test, Y_test):
    # Train the model
    log = model.fit(
        X_train, Y_train,
        epochs=100,
        batch_size=32,
        verbose=True,
        callbacks=[tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            min_delta=0,
            patience=10,
            verbose=1,
            restore_best_weights=True
        )],
        validation_data=(X_test, Y_test)
    )

    # Evaluate the model on the test set
    test_loss, test_acc = model.evaluate(X_test, Y_test)
    print("Test accuracy:", test_acc)

    return log
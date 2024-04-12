import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

def load_mnist_data():
    """Load and preprocess the MNIST dataset."""
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
    test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255
    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)
    return (train_images, train_labels), (test_images, test_labels)

def build_cnn_model():
    """Build a convolutional neural network (CNN) model."""
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_and_evaluate_model(model, train_images, train_labels, test_images, test_labels):
    """Train and evaluate the CNN model."""
    model.fit(train_images, train_labels, epochs=5, batch_size=64, validation_split=0.1)
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print(f'Test accuracy: {test_acc}')
    return model

def save_model(model, model_path):
    """Save the trained model to a file."""
    model.save(model_path)

def main():
    # Load MNIST dataset
    (train_images, train_labels), (test_images, test_labels) = load_mnist_data()

    # Build CNN model
    model = build_cnn_model()

    # Train and evaluate the model
    trained_model = train_and_evaluate_model(model, train_images, train_labels, test_images, test_labels)

    # Save the trained model
    model_path = 'pretrained_model.h5'
    save_model(trained_model, model_path)

if __name__ == '__main__':
    main()

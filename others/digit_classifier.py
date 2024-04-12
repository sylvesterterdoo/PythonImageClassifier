import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

def load_mnist_data():
    # Load the MNIST dataset
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

    # Preprocess the data
    train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
    test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255

    train_labels = tf.keras.utils.to_categorical(train_labels)
    test_labels = tf.keras.utils.to_categorical(test_labels)

    return (train_images, train_labels), (test_images, test_labels)

def build_cnn_model():
    # Define the CNN model architecture
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def train_and_save_model():
    # Load MNIST dataset
    (train_images, train_labels), (test_images, test_labels) = load_mnist_data()

    # Build CNN model
    model = build_cnn_model()

    # Train the model
    model.fit(train_images, train_labels, epochs=5, batch_size=64, validation_split=0.1)

    # Evaluate the model
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    st.write(f'Test accuracy: {test_acc}')

    # Save the trained model
    model.save('pretrained_model.h5')

def predict_digit(image):
    # Load the trained model
    model = load_model('pretrained_model.h5')

    # Preprocess the image for prediction
    image = image.convert('L').resize((28, 28))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Make prediction using the loaded model
    prediction = model.predict(image_array)
    predicted_digit = np.argmax(prediction)

    return predicted_digit

def main():
    st.title('Digit Classifier')
    st.write('Upload an image of a digit (0-9) to classify')

    # File uploader widget
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)

            # Predict the digit in the uploaded image
            predicted_digit = predict_digit(image)
            st.write(f'Predicted Digit: {predicted_digit}')

        except Exception as e:
            st.write(f"Error predicting digit: {e}")

if __name__ == '__main__':
    # Train and save the model before running the Streamlit app
    train_and_save_model()
    main()

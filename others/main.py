#import streamlit as st
#import tensorflow as tf
#from PIL import Image
#import numpy as np
#
## Load the pre-trained model
#model = tf.keras.models.load_model('pretrained_modelxx.h5')
#
## Function to preprocess the image for model prediction
#def preprocess_image(image):
#    # Resize the image to 28x28 and convert to grayscale
#    image = image.convert('L').resize((28, 28))
#    # Normalize the pixel values to range [0, 1]
#    image_array = np.array(image) / 255.0
#    # Expand dimensions to match model input shape (add batch dimension)
#    image_array = np.expand_dims(image_array, axis=0)
#    return image_array
#
## Function to predict the digit from the image
#def predict_digit(image):
#    # Preprocess the image
#    processed_image = preprocess_image(image)
#    # Make prediction using the loaded model
#    prediction = model.predict(processed_image)
#    # Get the predicted digit (index of the maximum probability)
#    predicted_digit = np.argmax(prediction)
#    return predicted_digit
#
## Streamlit app
#def main():
#    st.title('Digit Classifier')
#    st.write('Upload an image of a digit (0-9) to classify')
#
#    # File uploader widget
#    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
#
#    if uploaded_file is not None:
#        # Display the uploaded image
#        image = Image.open(uploaded_file)
#        st.image(image, caption='Uploaded Image', use_column_width=True)
#
#        # Classify the digit in the uploaded image
#        predicted_digit = predict_digit(image)
#        st.write(f'Predicted Digit: {predicted_digit}')
#
## Run the Streamlit app
#if __name__ == '__main__':
#    main()
#
#

import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Streamlit app
def main():
    st.title('Digit Classifier')
    st.write('Upload an image of a digit (0-9) to classify')

    # File uploader widget
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            # Load the pre-trained model
            model = tf.keras.models.load_model('pretrained_model.h5')

            # Function to preprocess the image for model prediction
            def preprocess_image(image):
                # Resize the image to 28x28 and convert to grayscale
                image = image.convert('L').resize((28, 28))
                # Normalize the pixel values to range [0, 1]
                image_array = np.array(image) / 255.0
                # Expand dimensions to match model input shape (add batch dimension)
                image_array = np.expand_dims(image_array, axis=0)
                return image_array

            # Function to predict the digit from the image
            def predict_digit(image):
                # Preprocess the image
                processed_image = preprocess_image(image)
                # Make prediction using the loaded model
                prediction = model.predict(processed_image)
                # Get the predicted digit (index of the maximum probability)
                predicted_digit = np.argmax(prediction)
                return predicted_digit

            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)

            # Classify the digit in the uploaded image
            predicted_digit = predict_digit(image)
            st.write(f'Predicted Digit: {predicted_digit}')

        except Exception as e:
            st.write(f"Error loading model: {e}")

# Run the Streamlit app
if __name__ == '__main__':
    main()


#import streamlit as st
#import tensorflow as tf
#from PIL import Image
#import numpy as np
#import requests
#
## Load the MobileNetV2 model
#model = tf.keras.applications.MobileNetV2(
#    weights='imagenet',  # Use pre-trained ImageNet weights
#    input_shape=(224, 224, 3),  # Expected input shape for MobileNetV2
#    include_top=True  # Include classification layer for 1000 ImageNet classes
#)
## Compile the model
#model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#
## Function to preprocess the image for model prediction
## def preprocess_image(image):
##     image = image.resize((224, 224))  # Resize image to expected input size for MobileNetV2
##     image_array = np.array(image)  # Convert PIL image to numpy array
##     image_array = image_array / 255.0  # Normalize pixel values to [0, 1]
##     image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
##     return image_array
#
#def preprocess_image(image):
#    image = image.convert('RGB')  # Convert image to RGB format
#    image = image.resize((224, 224))  # Resize image to expected input size for MobileNetV2
#    image_array = np.array(image)  # Convert PIL image to numpy array
#    image_array = image_array / 255.0  # Normalize pixel values to [0, 1]
#    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
#    return image_array
#
#
## Function to predict image class using the model
#def predict_image_class(image):
#    processed_image = preprocess_image(image)
#    prediction = model.predict(processed_image)
#    predicted_class = tf.keras.applications.imagenet_utils.decode_predictions(prediction, top=1)[0][0]
#    return predicted_class
#
## Streamlit app
#def main():
#    st.title('Image Classifier with MobileNetV2')
#
#    # File uploader widget
#    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
#
#    if uploaded_file is not None:
#        # Display the uploaded image
#        image = Image.open(uploaded_file)
#        st.image(image, caption='Uploaded Image', use_column_width=True)
#
#        # Classify the image
#        if st.button('Classify'):
#            with st.spinner('Classifying...'):
#                predicted_class = predict_image_class(image)
#                st.success(f'Predicted Class: {predicted_class[1]} (Confidence: {predicted_class[2]*100:.2f}%)')
#
## Run the Streamlit app
#if __name__ == '__main__':
#    main()
#
#

#import streamlit as st
#import tensorflow as tf
#from PIL import Image
#import numpy as np
#
## Load the MobileNetV2 model pre-trained on ImageNet (without the top classification layer)
#base_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
#
## Add a custom top layer for digit classification
#model = tf.keras.Sequential([
#    base_model,
#    tf.keras.layers.GlobalAveragePooling2D(),
#    tf.keras.layers.Dense(128, activation='relu'),
#    tf.keras.layers.Dropout(0.5),
#    tf.keras.layers.Dense(10, activation='softmax')  # Output layer for digits 0-9
#])
#
## Load the weights of the digit classifier
#model.load_weights('pretrained_model.h5')
#
## Function to preprocess the image for MobileNetV2 input
#def preprocess_image(image):
#    image = image.convert('RGB').resize((224, 224))
#    image_array = np.array(image)
#    image_array = tf.keras.applications.mobilenet_v2.preprocess_input(image_array)
#    return image_array
#
## Function to predict the digit (0-9) from the image
#def predict_digit(image):
#    processed_image = preprocess_image(image)
#    processed_image = np.expand_dims(processed_image, axis=0)
#    prediction = model.predict(processed_image)
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
#        try:
#            # Display the uploaded image
#            image = Image.open(uploaded_file)
#            st.image(image, caption='Uploaded Image', use_column_width=True)
#
#            # Predict the digit in the uploaded image
#            predicted_digit = predict_digit(image)
#            st.write(f'Predicted Digit: {predicted_digit}')
#
#        except Exception as e:
#            st.write(f"Error predicting digit: {e}")
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

# Function to preprocess the image for MobileNetV2 input
def preprocess_image(image):
    image = image.convert('RGB').resize((224, 224))
    image_array = np.array(image)
    image_array = tf.keras.applications.mobilenet_v2.preprocess_input(image_array)
    return image_array

# Load the MobileNetV2 model pre-trained on ImageNet (without the top classification layer)
base_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add a custom top layer for digit classification
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='softmax')  # Output layer for digits 0-9
])

# Streamlit app
def main():
    st.title('Digit Classifier')
    st.write('Upload an image of a digit (0-9) to classify')

    # File uploader widget
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            # Load the weights of the digit classifier
            model.load_weights('pretrained_model.h5')

            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)

            # Predict the digit in the uploaded image
            processed_image = preprocess_image(image)
            processed_image = np.expand_dims(processed_image, axis=0)
            prediction = model.predict(processed_image)
            predicted_digit = np.argmax(prediction)
            st.write(f'Predicted Digit: {predicted_digit}')

        except Exception as e:
            st.write(f"Error predicting digit: {e}")

# Run the Streamlit app
if __name__ == '__main__':
    main()


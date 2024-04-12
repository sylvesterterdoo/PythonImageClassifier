import streamlit as st
import requests
import matplotlib.pyplot as plt
import datetime
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

def get_cryptocurrency_price_history(crypto_name, days):
    # Base URL for CoinGecko API
    base_url = "https://api.coingecko.com/api/v3"

    # Endpoint to get cryptocurrency price history
    endpoint = f"/coins/{crypto_name}/market_chart"

    # Parameters to pass in the request
    params = {
        "vs_currency": "usd",      # Currency for price conversion (USD in this case)
        "days": days               # Number of days of historical data
    }

    try:
        # Make GET request to CoinGecko API
        response = requests.get(base_url + endpoint, params=params)
        response.raise_for_status()  # Raise an exception for bad response status

        # Parse response data as JSON
        data = response.json()

        if data:
            # Extract prices and timestamps from the response
            prices = [entry[1] for entry in data["prices"]]
            timestamps = [datetime.datetime.fromtimestamp(entry[0] / 1000) for entry in data["prices"]]

            return prices, timestamps
        else:
            return None, None

    except requests.exceptions.RequestException as e:
        st.write(f"Error occurred: {e}")
        return None, None

def plot_comparison_chart(prices1, timestamps1, prices2, timestamps2, crypto_name1, crypto_name2):
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot cryptocurrency price data
    if prices1 and timestamps1:
        ax.plot(timestamps1, prices1, label=f"{crypto_name1.capitalize()} Price (USD)")

    if prices2 and timestamps2:
        ax.plot(timestamps2, prices2, label=f"{crypto_name2.capitalize()} Price (USD)")

    # Set plot title, labels, and legend
    ax.set_title(f"{crypto_name1.capitalize()} vs {crypto_name2.capitalize()} Price Comparison")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.grid(True)
    ax.legend()

    return fig

# Main Streamlit app
def main():
    st.title("Cryptocurrency Price Comparison")

    # Sidebar selection for comparison
    page = st.sidebar.selectbox("Select Page", ["Single Coin", "Coin Comparison", "Image Classifier"])

    if page == "Single Coin":
        crypto_name = st.text_input("Enter cryptocurrency name (e.g., bitcoin, ethereum)")

        if crypto_name:
            days = st.sidebar.selectbox("Select timeframe (days)", [7, 30, 365], index=2)
            prices, timestamps = get_cryptocurrency_price_history(crypto_name.lower(), days)

            if prices and timestamps:
                # Calculate max and min prices
                max_price = max(prices)
                min_price = min(prices)
                max_timestamp = timestamps[prices.index(max_price)]
                min_timestamp = timestamps[prices.index(min_price)]

                # Print max and min prices along with corresponding timestamps
                st.write(f"Maximum price of {crypto_name.capitalize()} over the last {days} days: ${max_price:.2f} on {max_timestamp.date()}")
                st.write(f"Minimum price of {crypto_name.capitalize()} over the last {days} days: ${min_price:.2f} on {min_timestamp.date()}")

                # Plot cryptocurrency price history
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(timestamps, prices, label=f"{crypto_name.capitalize()} Price (USD)")
                ax.set_title(f"{crypto_name.capitalize()} Price Over the Last {days} Days")
                ax.set_xlabel("Date")
                ax.set_ylabel("Price (USD)")
                ax.grid(True)
                ax.legend()
                st.pyplot(fig)

    elif page == "Coin Comparison":
        crypto_name1 = st.text_input("Enter first cryptocurrency name (e.g., bitcoin, ethereum)")
        crypto_name2 = st.text_input("Enter second cryptocurrency name (e.g., bitcoin, ethereum)")

        if crypto_name1 and crypto_name2:
            days = st.sidebar.selectbox("Select timeframe (days)", [7, 30, 365], index=2)
            prices1, timestamps1 = get_cryptocurrency_price_history(crypto_name1.lower(), days)
            prices2, timestamps2 = get_cryptocurrency_price_history(crypto_name2.lower(), days)

            if prices1 and timestamps1 and prices2 and timestamps2:
                # Plot comparison chart
                fig = plot_comparison_chart(prices1, timestamps1, prices2, timestamps2, crypto_name1, crypto_name2)
                st.pyplot(fig)

    elif page == "Image Classifier":
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



if __name__ == "__main__":
    main()

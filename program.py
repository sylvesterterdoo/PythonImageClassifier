import streamlit as st
import requests
import matplotlib.pyplot as plt
import datetime
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
import time

# Function to retrieve cryptocurrency price history for a specific number of days
def get_cryptocurrency_price_history(crypto_name, days):
    base_url = "https://api.coingecko.com/api/v3"
    endpoint = f"/coins/{crypto_name}/market_chart"

    params = {
        "vs_currency": "usd",
        "days": days
    }

    try:
        response = requests.get(base_url + endpoint, params=params)
        response.raise_for_status()

        data = response.json()

        if data:
            prices = [entry[1] for entry in data["prices"]]
            timestamps = [datetime.datetime.fromtimestamp(entry[0] / 1000) for entry in data["prices"]]
            return prices, timestamps
        else:
            return None, None

    except requests.exceptions.RequestException as e:
        st.write(f"Error occurred: {e}")
        return None, None

# Function to retrieve cryptocurrency price history for five years (multiple requests)
#def get_cryptocurrency_price_history_five_years(crypto_name, days_per_request, num_years=5):
#    all_prices = []
#    all_timestamps = []
#
#    for _ in range(num_years):
#        prices, timestamps = get_cryptocurrency_price_history(crypto_name, days_per_request)
#        if prices and timestamps:
#            all_prices.extend(prices)
#            all_timestamps.extend(timestamps)
#
#    return all_prices, all_timestamps
#
def get_cryptocurrency_price_history_five_years(crypto_name, days_per_request, num_years=5):
    all_prices = []
    all_timestamps = []

    for _ in range(num_years):
        prices, timestamps = get_cryptocurrency_price_history(crypto_name, days_per_request)
        if prices and timestamps:
            all_prices.extend(prices)
            all_timestamps.extend(timestamps)

        # Introduce a delay of 1 second between consecutive API requests
        time.sleep(1)

    return all_prices, all_timestamps

# Function to plot comparison chart for cryptocurrency prices
def plot_comparison_chart(prices1, timestamps1, prices2, timestamps2, crypto_name1, crypto_name2):
    fig, ax = plt.subplots(figsize=(12, 6))

    if prices1 and timestamps1:
        ax.plot(timestamps1, prices1, label=f"{crypto_name1.capitalize()} Price (USD)")

    if prices2 and timestamps2:
        ax.plot(timestamps2, prices2, label=f"{crypto_name2.capitalize()} Price (USD)")

    ax.set_title(f"{crypto_name1.capitalize()} vs {crypto_name2.capitalize()} Price Comparison")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.grid(True)
    ax.legend()

    return fig

# Function to predict digit from uploaded image
def predict_digit(image):
    model = load_model('numberclassifier.keras')

    if image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info):
        r, g, b, a = image.split()
    else:
        image = image.convert('RGBA')
        r, g, b, a = image.split()

    alpha_image = Image.merge('L', (a,))
    alpha_image = alpha_image.resize((28, 28))
    image_array = np.array(alpha_image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    prediction = model.predict(image_array)
    predicted_digit = np.argmax(prediction)

    return predicted_digit

def main():
    st.title("Cryptocurrency and Image Classifier App")

    page = st.sidebar.selectbox("Select Page", ["Single Coin", "Coin Comparison", "Image Classifier"])

    if page == "Single Coin":
        st.header("Cryptocurrency Price Analysis")
        crypto_name = st.text_input("Enter cryptocurrency name (e.g., bitcoin, ethereum)")

        if crypto_name:
            days = st.sidebar.selectbox("Select timeframe (days)", [30, 365], index=1)
            prices, timestamps = get_cryptocurrency_price_history(crypto_name.lower(), days)

            if prices and timestamps:
                max_price = max(prices)
                min_price = min(prices)
                max_timestamp = timestamps[prices.index(max_price)]
                min_timestamp = timestamps[prices.index(min_price)]

                st.write(f"Maximum price of {crypto_name.capitalize()} over the last {days} days: ${max_price:.2f} on {max_timestamp.date()}")
                st.write(f"Minimum price of {crypto_name.capitalize()} over the last {days} days: ${min_price:.2f} on {min_timestamp.date()}")

                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(timestamps, prices, label=f"{crypto_name.capitalize()} Price (USD)")
                ax.set_title(f"{crypto_name.capitalize()} Price Over the Last {days} Days")
                ax.set_xlabel("Date")
                ax.set_ylabel("Price (USD)")
                ax.grid(True)
                ax.legend()
                st.pyplot(fig)

    elif page == "Coin Comparison":
        st.header("Cryptocurrency Price Comparison")
        crypto_name1 = st.text_input("Enter first cryptocurrency name (e.g., bitcoin, ethereum)")
        crypto_name2 = st.text_input("Enter second cryptocurrency name (e.g., bitcoin, ethereum)")

        if crypto_name1 and crypto_name2:
            days = st.sidebar.selectbox("Select timeframe (days)", [7, 30, 365, 1825], index=2)
            if days == 1825:
                prices1, timestamps1 = get_cryptocurrency_price_history_five_years(crypto_name1.lower(), 100, num_years=5)
                prices2, timestamps2 = get_cryptocurrency_price_history_five_years(crypto_name2.lower(), 100, num_years=5)
            else:
                prices1, timestamps1 = get_cryptocurrency_price_history(crypto_name1.lower(), days)
                prices2, timestamps2 = get_cryptocurrency_price_history(crypto_name2.lower(), days)

            if prices1 and timestamps1 and prices2 and timestamps2:
                fig = plot_comparison_chart(prices1, timestamps1, prices2, timestamps2, crypto_name1, crypto_name2)
                st.pyplot(fig)

    elif page == "Image Classifier":
        st.title('Digit Classifier')
        st.write('Upload an image of a digit (0-9) to classify')

        uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file)
                st.image(image, caption='Uploaded Image', use_column_width=True)

                predicted_digit = predict_digit(image)
                st.write(f'Predicted Digit: {predicted_digit}')

            except Exception as e:
                st.write(f"Error predicting digit: {e}")

if __name__ == "__main__":
    main()

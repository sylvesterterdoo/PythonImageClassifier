import streamlit as st
import requests
import matplotlib.pyplot as plt
import datetime

def get_cryptocurrency_price_history(crypto_name):
    # Base URL for CoinGecko API
    base_url = "https://api.coingecko.com/api/v3"

    # Endpoint to get cryptocurrency price history
    endpoint = f"/coins/{crypto_name}/market_chart"

    # Parameters to pass in the request
    params = {
        "vs_currency": "usd",      # Currency for price conversion (USD in this case)
        "days": 365                # Number of days of historical data (1 year)
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

            # Calculate max and min prices
            max_price = max(prices)
            min_price = min(prices)

            # Find corresponding timestamps for max and min prices
            max_timestamp = timestamps[prices.index(max_price)]
            min_timestamp = timestamps[prices.index(min_price)]

            # Print max and min prices along with corresponding timestamps
            st.write(f"Maximum price of {crypto_name.capitalize()} over the last year: ${max_price:.2f} on {max_timestamp.date()}")
            st.write(f"Minimum price of {crypto_name.capitalize()} over the last year: ${min_price:.2f} on {min_timestamp.date()}")

            # Plot the cryptocurrency price history
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(timestamps, prices, label=f"{crypto_name.capitalize()} Price (USD)")
            ax.set_title(f"{crypto_name.capitalize()} Price Over the Last Year")
            ax.set_xlabel("Date")
            ax.set_ylabel("Price (USD)")
            ax.grid(True)
            ax.legend()
            st.pyplot(fig)
        else:
            st.write("No data returned from the API.")

    except requests.exceptions.RequestException as e:
        st.write(f"Error occurred: {e}")

# Main Streamlit app
def main():
    st.title("Cryptocurrency Price History Viewer")

    # Input for cryptocurrency name
    crypto_name = st.text_input("Enter cryptocurrency name (e.g., bitcoin, ethereum)")

    if crypto_name:
        # Call function to get cryptocurrency price history and plot
        get_cryptocurrency_price_history(crypto_name.lower())

if __name__ == "__main__":
    main()

import requests
import json

# Configuration for the A2A agent server
AGENT_URL = "http://127.0.0.1:8002"
INVOKE_PATH = "/invoke"

def get_a2a_short_term_prediction(ticker: str) -> dict:
    """Gets a short-term price prediction for a single stock ticker. Use for specific queries like 'predict MSFT' or 'forecast for GOOG'."""
    if not ticker or not isinstance(ticker, str):
        return {"error": "Invalid ticker provided. Must be a non-empty string."}

    full_url = f"{AGENT_URL}{INVOKE_PATH}"
    payload = {
        "action": "short_term_predict",
        "kwargs": {"ticker": ticker.upper()}
    }
    
    print(f"Calling A2A prediction agent for {ticker} at {full_url}...")

    try:
        response = requests.post(full_url, json=payload, timeout=15)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        
        # The server might return JSON with a 'result' or just the raw prediction
        response_data = response.json()
        if "result" in response_data:
            return response_data["result"]
        elif "output" in response_data:
            return response_data["output"]
        else:
            return response_data # Return the whole thing if structure is unknown

    except requests.exceptions.ConnectionError:
        return {"error": f"Connection failed. Is the agent server running at {AGENT_URL}?"}
    except requests.exceptions.HTTPError as e:
        return {"error": f"HTTP error occurred: {e}. Server response: {e.response.text}"}
    except requests.exceptions.RequestException as e:
        return {"error": f"An unexpected error occurred: {e}"}
    except json.JSONDecodeError:
        return {"error": f"Failed to decode JSON from server response. Response text: {response.text}"}

if __name__ == '__main__':
    # Example of how to use the function directly for testing
    test_ticker = "NVDA"
    prediction = get_a2a_short_term_prediction(test_ticker)
    print(f"\nPrediction result for {test_ticker}:")
    print(prediction)
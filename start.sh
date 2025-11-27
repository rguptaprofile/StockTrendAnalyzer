#!/bin/bash
# This script starts the backend agent and the Streamlit UI.

# Start the StockForcasterAgent API server in the background.
# The '&' symbol runs the command as a background process.
echo "Starting Stock Forcaster Agent in the background..."
python /app/prediAgent/short_Term_Stock_Predictor/StockForcasterAgent.py &

# Wait for a few seconds to ensure the background server has time to initialize.
sleep 5

# Start the Streamlit UI in the foreground.
# This will be the main process for the container.
echo "Starting Streamlit UI in the foreground..."
streamlit run /app/prediAgent/trading_bot/Stocktrender_Chatbot_UI.py --server.port=8501 --server.address=0.0.0.0

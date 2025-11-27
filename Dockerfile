# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*


# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# --no-cache-dir makes the image smaller
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code into the container
COPY . .

# Copy the startup script into the container
COPY start.sh .

# Make the startup script executable
RUN chmod +x ./start.sh

# Expose the port that Streamlit runs on
EXPOSE 8501

# Set the command to run the startup script
CMD ["./start.sh"]
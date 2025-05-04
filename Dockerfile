FROM python:3.13-slim

# Install system dependencies
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy the entire application into the container
COPY . /app

# Change to the backend directory
WORKDIR /app/backend

# Install Python dependencies
RUN pip install -r requirements.txt
RUN pip install gunicorn

# Expose the application port
EXPOSE 5000

# Start the Gunicorn server with optimized settings
CMD ["gunicorn", "Model:app", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "120"]
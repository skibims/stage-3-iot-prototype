FROM python:3.x-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy your application code into the container
COPY . /app/.

# Navigate to your backend directory
WORKDIR /app/backend

# Install your Python dependencies
RUN pip install -r requirements.txt

# Define the command to run your application
CMD ["python", "Model.py"]
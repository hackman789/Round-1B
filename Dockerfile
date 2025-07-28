# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the dependencies file to the working directory
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# This step downloads the libraries and the AI model, so the final container runs offline.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the main application script into the container at /app
COPY main.py .

# Create a directory for the PDFs inside the container
RUN mkdir pdfs

# Define the entry point for the container. This makes the container executable.
# It will run "python main.py" with any arguments provided to `docker run`.
ENTRYPOINT ["python", "main.py"]
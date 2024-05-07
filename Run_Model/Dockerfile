# Use a Python base image
FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Copy application files to the container
COPY inference_script.py requirements.txt my_model.h5 /app/

COPY scaler.pkl /app/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port that the Flask app runs on
EXPOSE 5000

# Command to run the application
CMD ["python", "inference_script.py"]
# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container
COPY . .

# Make port 8888 available if you want to run Jupyter Notebook (optional)
# EXPOSE 8888

# Define environment variable
ENV PYTHONUNBUFFERED=1

# Run the command when the container launches
CMD ["python", "audio_analysis.py"]
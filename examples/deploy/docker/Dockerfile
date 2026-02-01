# Use an official Python 3.11 runtime as a parent image
FROM python:3.11.7-slim

# Set the working directory in the container
WORKDIR /usr/src/app

# Install KlongPy
RUN pip install "klongpy[all]"

# Make port available to the world outside this container (adjust if needed)
EXPOSE 8000

# Run app.py when the container launches, ensuring it uses the KG_FILE_PATH
CMD kgpy $KG_FILE_PATH
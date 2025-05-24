# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the entire project directory into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make ports 5000 (API) and 5001 (Webapp) available to the world outside this container
EXPOSE 5000
EXPOSE 5001

# Run the API and Webapp when the container launches
# Using a simple script to start both
COPY scripts/start_servers.sh /app/scripts/
RUN chmod +x /app/scripts/start_servers.sh
CMD ["/app/scripts/start_servers.sh"]
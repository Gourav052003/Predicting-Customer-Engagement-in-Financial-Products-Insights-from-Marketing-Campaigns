FROM python:3.9.7

# Set the working directory
WORKDIR /application

# Copy the application code
COPY . /application/

# Expose the application port
EXPOSE 8501 8502

# Install Python dependencies
RUN pip install -r requirements.txt

# Set the command to run the application
CMD ["supervisord", "-c", "/application/supervisord.conf"]

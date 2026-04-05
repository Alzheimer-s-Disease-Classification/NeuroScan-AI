# Use a standard Python environment
FROM python:3.10

# Set the working directory inside the server
WORKDIR /app

# Copy the requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all your project files into the server
COPY . .

# Expose the Hugging Face port
EXPOSE 7860

# The command to start your app
CMD ["python", "app.py"]
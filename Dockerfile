FROM tensorflow/tensorflow:2.20.0

WORKDIR /app

# Copy only the requirements first
COPY requirements.txt .

# Install dependencies while bypassing the system-level distutils conflict
RUN pip install --no-cache-dir --ignore-installed blinker -r requirements.txt

# Copy the rest of your files (including the patched app.py and your .h5 model)
COPY . .

# Open the port for the API
EXPOSE 8000

# Run the Flask app
CMD ["python", "-u", "app.py"]
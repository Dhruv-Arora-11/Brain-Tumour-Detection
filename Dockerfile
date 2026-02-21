# Use official TensorFlow CPU image as base
FROM tensorflow/tensorflow:2.15.0
FROM python:3.11
# Set working directory inside container
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt --ignore-installed
COPY model_training.py .
COPY predicting_single_image.py .
COPY trained_model.h5 .
COPY t1.jpg t2.jpg t3.jpg t4.jpg ./
COPY app.py .


# Default command: run model training
CMD ["python", "app.py"]



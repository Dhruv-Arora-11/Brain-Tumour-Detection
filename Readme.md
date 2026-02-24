🧠 Brain Tumor Detection API (CNN + Docker)

A Convolutional Neural Network (CNN) based brain tumor classification system built using TensorFlow/Keras and deployed as a Dockerized Flask API.
 
📌 Overview 

This project provides:

CNN model trained for brain tumor classification

REST API built using Flask

Fully Dockerized deployment

Ready for frontend or CI/CD integration

🛠 Tech Stack

Python 3.11

TensorFlow / Keras

Flask

Docker

📁 Project Structure
├── app.py                     # Flask API
├── model_training.py          # Model training script
├── predicting_single_image.py # Local prediction script
├── trained_model.h5           # Saved trained model
├── requirements.txt           # Dependencies
├── Dockerfile                 # Docker configuration
├── docker-compose.yml         # Optional compose setup
└── test_images/               # Sample images (optional)
🐳 Running with Docker (Recommended)
1️⃣ Build Docker Image

From the project root directory:

docker build -t brain_tumor_detection .
2️⃣ Run the Container
docker run -p 8000:8000 brain_tumor_detection

If successful, you will see:

Running on http://127.0.0.1:8000

The API is now live.

📡 API Usage
Endpoint
POST /predict
Full URL
http://127.0.0.1:8000/predict
🧪 Testing the API
🔹 Using Postman

Select POST

URL:

http://127.0.0.1:8000/predict

Go to Body → form-data

Add:

Key	Type	Value
file	File	Upload image

Click Send

Example Response
{
  "prediction": 1
}
🔹 Using curl
curl -X POST -F "file=@t1.jpg" http://127.0.0.1:8000/predict
🔄 Rebuild After Code Changes
docker build --no-cache -t brain_tumor_detection .
⚠ Important Notes

Model expects image size: 224 × 224

Input must be RGB image

API runs on port 8000

This uses Flask development server (not production-grade)

🚀 Future Improvements

Add Gunicorn for production

Add CI/CD pipeline

Connect MERN frontend

Deploy on cloud (AWS / Azure / GCP)
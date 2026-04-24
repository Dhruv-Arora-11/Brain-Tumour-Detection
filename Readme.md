# 🧠 Brain Tumor Detection API (CNN + Docker)

A Convolutional Neural Network (CNN) based brain tumor classification system built using TensorFlow/Keras and deployed as a Dockerized Flask API.

## 📌 Overview 

This project provides:
- **CNN model** trained for brain tumor classification.
- **REST API** built using Flask for real-time predictions.
- **Fully Dockerized** deployment for consistency across environments.
- Ready for frontend or CI/CD integration.

## 🛠 Tech Stack

- **Language:** Python 3.11
- **Frameworks:** TensorFlow / Keras, Flask
- **Containerization:** Docker & Docker Compose


## 📁 Project Structure

```text
├── app.py                            # Flask API
├── model_training_transfer_learning.py # Model training script (Transfer Learning)
├── predicting_single_image.py        # Local prediction script
├── preprocessing.py                  # Data preprocessing utilities
├── trained_model.keras               # Saved trained model
├── requirements.txt                  # Dependencies
├── Dockerfile                        # Docker configuration
├── docker-compose.yml                # Docker Compose setup
├── model_training/                   # Directory for training logs/data
├── frontend/                         # Frontend application files
├── notebooks/                        # Jupyter notebooks
└── t1.jpg, t2.jpg...                # Sample test images
```

## 🐳 Running with Docker (Recommended)

### 1️⃣ Build Docker Image

From the project root directory:

```bash
docker build -t brain_tumor_detection .
```

### 2️⃣ Run the Container

```bash
docker run -p 8000:8000 brain_tumor_detection
```

If successful, you will see:

`Running on http://127.0.0.1:8000`

The API is now live.

## 📡 API Usage

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/predict` | Predicts whether an image contains a tumor |

**Full URL:** `http://127.0.0.1:8000/predict`

## 🧪 Testing the API

### 🔹 Using Postman

1. Select **POST** method.
2. Enter URL: `http://127.0.0.1:8000/predict`
3. Go to **Body** → **form-data**.
4. Add a key named `file`, set type to **File**, and upload an image.
5. Click **Send**.

**Example Response:**
```json
{
  "prediction": 1
}
```

### 🔹 Using curl

```bash
curl -X POST -F "file=@t1.jpg" http://127.0.0.1:8000/predict
```

## 🔄 Rebuild After Code Changes

```bash
docker build --no-cache -t brain_tumor_detection .
```

## ⚠ Important Notes

- **Model Input:** Expects image size of `224 × 224`.
- **Image Format:** Input must be an RGB image.
- **Port:** API runs on port `8000`.
- **Environment:** This uses the Flask development server (not recommended for production).

## 🚀 Future Improvements

- [ ] Add **Gunicorn/Nginx** for production-grade serving.
- [ ] Implement a **CI/CD pipeline** (GitHub Actions).
- [ ] Connect a **MERN stack frontend**.
- [ ] Deploy to cloud platforms (**AWS / Azure / GCP**).

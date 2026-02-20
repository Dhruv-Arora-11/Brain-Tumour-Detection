# Capstone Project — Image Classification with CNN

A CNN-based image classifier built with TensorFlow/Keras, fully Dockerized for easy setup.

## Prerequisites 

- [Docker](https://docs.docker.com/get-docker/) installed on your machine

## Project Structure

```
├── model_training.py         # Train the CNN model
├── predicting_single_image.py # Predict on a single image
├── model.ipynb               # Jupyter notebook (development)
├── trained_model.h5          # Pre-trained model weights
├── requirements.txt          # Python dependencies
├── Dockerfile
├── docker-compose.yml
├── Training/                 # Training dataset (not in repo — add your own)
└── Testing/                  # Testing dataset  (not in repo — add your own)
```

## Running with Docker

### 1. Add your datasets

Place your `Training/` and `Testing/` folders in the project root. They should contain sub-folders for each class.

### 2. Train the model

```bash
docker compose up train
```

The trained model will be saved to the `output/` folder on your host machine.

### 3. Run prediction

```bash
docker compose up predict
```

### Rebuild after code changes

```bash
docker compose build
```

## Running without Docker

```bash
pip install -r requirements.txt
python model_training.py
python predicting_single_image.py
```

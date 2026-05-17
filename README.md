# Assignment 7 part 1 — FashionMNIST CNN Deployment (FastAPI + Docker)

## Overview
This project implements a Convolutional Neural Network (CNN) trained on the FashionMNIST dataset and deployed using a FastAPI web service. The model classifies grayscale 28×28 images into 10 clothing categories.

---

## Tech Stack
- Python
- PyTorch
- FastAPI
- Docker
- Uvicorn

---

## Model
The CNN consists of:
- 3 convolutional layers
- Batch normalization
- ReLU activation
- Max pooling
- Fully connected classifier

The model is trained on FashionMNIST and saved as `model.pt`.

---

## API Endpoints

### Health Check

GET /health


### Prediction

POST /predict


Upload a 28×28 grayscale image and receive:
- predicted class
- confidence score
- probability distribution

---

## Run Locally

### Install dependencies

pip install -r requirements.txt


### Train model

python train.py


### Run API

uvicorn app:app --reload


---

## Docker Run

### Build image

docker build -t fashionmnist-api .


### Run container

docker run -p 8000:8000 fashionmnist-api


---

## Port Mapping

localhost:8000 → container:8000


---

## Example Output

Predicted Class: Sneaker
Confidence: 0.99

---

## Author
Deep Learning Deployment Project

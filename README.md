Potato Leaf Disease Prediction

A deep learning–based web application that predicts diseases in potato leaves from images using a trained CNN model.

Table of Contents
    Overview
    Features
    Tech Stack
    Installation
    Usage
    Dataset
    Model
    Screenshots
    License

1) Overview

Potato crops are vulnerable to various diseases that reduce yield and quality. This project uses deep learning (CNN) to detect and classify diseases in potato leaves such as:

* Early blight
* Late blight
* Healthy

The model is trained on the PlantVillage dataset and deployed as a web app with a frontend and backend.

2)Features
    Upload a potato leaf image and get instant disease prediction.
    Clean UI built with HTML/CSS/JS (or React if applicable).
    Trained deep learning model (TensorFlow/Keras).
    REST API backend using Flask or FastAPI.
    Model saving/loading with `.h5` or `.pb`.

3)Tech Stack
    | Area       | Tools/Frameworks                       |
    | ---------- | -------------------------------------- |
    | Language   | Python                                 |
    | Model      | TensorFlow / Keras                     |
    | Backend    | Flask / FastAPI                        |
    | Frontend   | HTML / CSS / JavaScript (or React)     |
    | Deployment | GitHub / Localhost / Heroku (optional) |


4)Installation:
    bash
    git clone https://github.com/danyaldany/potato_leaf_disease_prediction.git
    cd potato_leaf_disease_prediction
    pip install -r requirements.txt

5)Usage
    1. Start backend server
    bash
       python api/app.py
    2. Open frontend:
       Open `frontend/index.html` in your browser.
    3. Upload Image:
       Upload a potato leaf image and get the predicted disease.

6)Dataset:
    Used: PlantVillage Dataset: (https://www.kaggle.com/datasets/emmarex/plantdisease)
    Classes:
      Early Blight
      Late Blight
      Healthy

7)Model
    CNN trained on image data
    Accuracy: 0.9583
    Saved model: `saved_model/model.h5`


8)License
    This project is open-source under the [MIT License](LICENSE).

# 🚗 Auto Insurance Claim Prediction

[![Live API on Render](https://img.shields.io/badge/Live%20API%20Deployment-Render-blue?style=for-the-badge&logo=render)](https://auto-insurance-claim-prediction.onrender.com/docs)
![Deployed Service Screenshot](https://github.com/vandithavb/machine-learning-zoomcamp-2025/blob/main/Auto_Insurance_Claim_Prediction/deployed%20model%20on%20render.png?raw=true)

This project focuses on building, optimizing, and deploying a machine learning model to predict whether an individual will file an insurance claim, based on various driver, vehicle, and demographic features. The goal is to assist insurance companies in better risk assessment and premium optimization, thus addressing a key challenge in insurance risk management.

## 🎯 Project Goal (Description of the Problem)

The primary objective is to develop a highly accurate classification model that can predict the binary outcome (Claim Filed: **Yes** or **No**). This prediction is crucial for:

  * **Risk Mitigation:** Identifying high-risk profiles to adjust policy terms.
  * **Operational Efficiency:** Streamlining claim processing and reducing potential fraud investigation costs.
  * **Pricing Strategy:** Ensuring fair and competitive premium setting based on calculated risk.
    
## ✅ **Project Summary (Quick Overview)**

- Downloaded Kaggle dataset  
- Performed extensive data cleaning & encoding  
- Conducted EDA (distributions, correlations, feature importance)  
- Trained 3 models: Logistic Regression, Decision Tree, Random Forest  
- Parameter
- Selected Random Forest as best model  
- Exported complete training pipeline into `train.py`  
- Saved final model as `model.bin`  
- Built FastAPI inference server (`app/predict.py`)  
- Created Dockerfile for containerization  
- Used `uv` for virtual environment  
- Deployed Dockerized API to Render.com  
- Added sample JSON inputs & screenshot in README

  
## 💾 Dataset

The model was trained using the **Car Insurance Claim Data** dataset.

  * **Source:** [Car Insurance Claim Data on Kaggle](https://www.kaggle.com/datasets/xiaomengsun/car-insurance-claim-data?resource=download)
  * **Data Download Instructions:** Download the dataset directly from the source link provided above. The main notebook assumes the data is available locally for processing.

## 🧑‍💻 Methodology & Model Selection (Notebook Content)

The project followed a standard Machine Learning workflow, with all analysis and selection detailed in the main Jupyter Notebook (`notebook.ipynb`):

1.  **Data Preparation and Cleaning:** Handling missing values, encoding categorical features, and preparing the dataset for modeling.
2.  **EDA, Feature Importance:** Exploratory Data Analysis was performed, followed by techniques like correlation analysis and **Feature Importance** plots to guide feature selection.
3.  **Model Selection and Tuning:** We trained and evaluated three models: Logistic Regression, Decision Tree, and Random Forest. **Random Forest Classifier** was selected due to its superior performance, specifically its high **ROC AUC score**, which is vital for a risk prediction problem. Hyperparameters were tuned to maximize this performance metric.

## ⚙️ Model Persistence (`train.py`)

The final, optimized Random Forest model is trained on the entire processed dataset using the best parameters identified during tuning.

  * **Script:** `train.py`
  * **Action:** Trains the model and uses the standard Python **`pickle`** library to serialize and save the model object as **`model.bin`**. This file acts as the artifact used for serving predictions.

## 🚀 Deployment Architecture and Web Service (`predict.py`, `Dockerfile`)

The trained model is deployed as a production-ready microservice using FastAPI and Docker.

  * **Web Service (`predict.py`): ** loads the saved **`model.bin`** and exposes a secure prediction endpoint using fast api.
  * **Containerization (`Dockerfile`):** A **`Dockerfile`** is included to build a portable image containing the Python environment, dependencies (listed in **`requirements.txt`**), and the FastAPI application, ensuring consistent execution across environments.
  * **Deployment (Render.com):** The Docker image is deployed to **Render.com**, providing a publicly accessible prediction API.


## 🚀 Deployment Architecture (FastAPI + Docker + Render)

### ✔ Inference API (`predict.py`)
- Loads `model.bin`
- Defines `/predict` endpoint
- Runs using **Uvicorn**
- Accepts JSON input and returns prediction + probability

### ✔ Containerization (`Dockerfile`)
- Python 3.9 base
- Installs dependencies from `requirements.txt`
- Runs FastAPI app via Uvicorn
- Makes deployment reproducible & portable

### ✔ Cloud Deployment (Render.com)
The Dockerized API is deployed to Render.

### 🔗 **Live API Documentation**
**https://auto-insurance-claim.onrender.com/docs**



## 🛠️ Local Setup and Run (Instructions)

To run this project locally, follow the instructions below.

### Prerequisites

You will need the following installed:

  * Python 3.8+
  * `uv` (used for virtual environment and dependency management)
  * Docker (Optional, for running the containerized version)

### Sample Input
```json
{
  "KIDSDRIV": 0,
  "AGE": 45,
  "HOMEKIDS": 1,
  "YOJ": 12,
  "INCOME": 95000,
  "HOME_VAL": 350000,
  "PARENT1": "No",
  "MSTATUS": "Z_Single",
  "GENDER": "M",
  "EDUCATION": "Bachelors",
  "OCCUPATION": "Professional",
  "TRAVTIME": 25,
  "CAR_USE": "Private",
  "BLUEBOOK": 18000,
  "TIF": 5,
  "CAR_TYPE": "Sedan",
  "RED_CAR": "no",
  "OLDCLAIM": 0,
  "REVOKED": "No",
  "MVR_PTS": 1,
  "CAR_AGE": 6,
  "URBANICITY": "Highly Urban/ Urban"
}
```

### 1. Installation and Environment

1.  **Clone the repository:**

    ```bash
    git clone [https://github.com/vandithavb/machine-learning-zoomcamp-2025](https://github.com/vandithavb/machine-learning-zoomcamp-2025)
    cd machine-learning-zoomcamp-2025/Auto_Insurance_Claim_Prediction
    ```


### 2. Training the Model (`train.py`)

Run the training script to generate the **`model.bin`** file, which is necessary for the API to run.

```bash
python train.py
```
### 3. Create Virtual Environment & Install Dependencies

This project uses a `requirements.txt` file for dependency management and a virtual environment created with **uv**.

#### 1️⃣ Create the virtual environment
```bash
uv venv
```
2️⃣ Activate the environment

Mac/Linux
```bash
source .venv/bin/activate
```
Windows
```bash
.\.venv\Scripts\activate
```
3️⃣ Install all dependencies
```bash
pip install -r requirements.txt
```
#### Running the FastAPI Application Locally
```bash
uvicorn predict:app --host 0.0.0.0 --port 9696 --reload
```

Once the server starts, open the UI

### Docker Containerization
You can also run the service inside a Docker container for a fully reproducible environment.
```bash
docker build -t auto-insurance-claim-model .
docker run -p 9696:9696 auto-insurance-claim-model
http://127.0.0.1:9696/docs
```
### 6. Cloud Deployment (Render.com)

This project is deployed as a **Dockerized FastAPI service** on Render.

#### 🌐 Live API URL
- **Swagger UI:** https://auto-insurance-claim-prediction.onrender.com/docs  
- **Root URL:** https://auto-insurance-claim-prediction.onrender.com/

---

### 🚀 Deployment Steps

1. Pushed project files (`predict.py`, `model.bin`, `Dockerfile`, `requirements.txt`) to GitHub  
2. Logged in to Render → **New Web Service**  
3. Selected this GitHub repo  
4. Chose **Docker** as the runtime  
5. Render auto-built the image using `Dockerfile`  
6. Service deployed with a public prediction endpoint  

You can test the `/predict` endpoint directly from the Swagger UI using the sample JSON provided in this README.




## Loan Default Analytics

## AI-Powered Loan Default Risk Prediction

Loan Default Analytics is a full-stack machine learning application that uses Logistic Regression classification to predict whether a borrower is likely to default on a loan. It combines a Python backend with FastAPI and an interactive React and TypeScript frontend to support credit risk assessment and data-driven lending analysis.

### Features

Default Risk Prediction: Classify loan applications based on predicted default risk.

Interactive Prediction Interface: Enter borrower and loan details through the frontend.

Analytics Dashboard: Explore loan and default-related insights.

FastAPI Backend: Handle prediction requests through REST API endpoints.

Model Persistence: Save and load the trained classification model using Joblib.

Model Training: Train the Logistic Regression classifier using historical loan data.

API Testing: Test backend endpoints and prediction responses.

### Tech Stack

#### Backend

Python

FastAPI

Scikit-learn

Joblib

Logistic Regression

#### Frontend

React

TypeScript

Vite

CSS

Machine Learning

Classification

Data preprocessing

Model evaluation

Loan Default risk prediction

### Project Structure

Loan-Default-Analytics/

├── backend/

│   ├── main.py
│   ├── app.py
│   ├── train.py
│   └── test_api.py

│
├── frontend/

│   └── src/
│       ├── components/
│       ├── PredictPage.tsx
│       ├── AnalysisPage.tsx
│       └── style.css

│
└── README.md

#### How It Works

The user enters borrower and loan information through the frontend.
The React application sends the input to the FastAPI backend.
The backend loads the trained Logistic Regression model using Joblib.
The model classifies the application and can estimate default probability if probability prediction is implemented.
The prediction result is returned to the frontend and displayed to the user.

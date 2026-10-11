# RetailPulse — AI-Powered Retail Analytics

Interactive retail analytics dashboard with customer segmentation, daily revenue forecasting, and inventory recommendations.

## Features
- Data cleaning and preprocessing
- Sales KPI analysis
- RFM customer segmentation using K-Means
- Daily revenue forecasting using Random Forest
- Inventory reorder recommendations
- Interactive Streamlit dashboard
- Automated tests using Pytest

## Tech Stack
Python, Pandas, NumPy, Scikit-learn, Plotly, Streamlit, Joblib, Pytest

## Run Locally
Install dependencies:

    python -m pip install -r requirements.txt

Start the dashboard:

    python -m streamlit run dashboard/app.py

Run tests:

    python -m pytest tests/test_retailpulse.py -v

## Limitations
Forecast metrics are based on historical test data. Inventory recommendations use assumed stock availability and supplier lead time. Customer cluster IDs are labels, not rankings.

## Project Status
Data science internship project.

## Live Demo

Try the deployed application: https://retailpulse-ispeikkn3i76p.streamlit.app/


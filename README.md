# ML-Comparison-Dashboard
This project is a Python Streamlit dashboard designed to compare the performance of machine learning classification models. It can use Logistic Regression, Random Forest, and Support Vector Machine on user-uploaded datasets. A user can select a target column, choose to encode categorical features, and scale inputs. The results are visualized on Plotly charts and the accuracy and precision are displayed.

[Live Demo](https://ml-comparison-dashboard.streamlit.app/)

## Features
- Dynamic Dataset Upload - Upload any CSV dataset and select a target column for classification.
- Model Selection - Choose which model(s) to use with customizable test sizes
- Preprocessing - Supports basic categorical encoding and feature scaling
- Displays accuracy and precision for each model.
- Visualizations:
  - Confusion Matrices
  - Feature importance plots for Random Forest
  - Bar charts for comparing model performance

## Tech Stack
- Python 3
- pandas, numpy - data preprocessing and manipulation
- scikit-learn - machine learning models
- Streamlit - web dashboard
- Plotly - visualizations

## Usage
1. Upload a dataset
2. Select target column, models, adjust test size, and toggle encoding or feature scaling
3. View results

## Example: Stock Price Prediction
Use generate_stock_data.py to get the historical data of AAPL, clean the data, add more feature columns, and put it into a csv.

Upload the CSV to the dashboard and select 'Price Up/Down' as the target column and select your models.

View the results across the different models 

## About
Developed by Evan Gaul as a portfolio project.

## License
MIT License

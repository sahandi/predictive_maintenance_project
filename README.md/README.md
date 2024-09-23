# Predictive Maintenance for Industrial Equipment

## Project Overview
This project focuses on predicting the Remaining Useful Life (RUL) of industrial equipment (e.g., turbofan engines) using sensor data. By predicting when equipment will fail, maintenance can be scheduled to avoid downtime.

## Project Structure
- `data/`: Contains the dataset.
- `models/`: Contains the saved model after training.
- `notebooks/`: Jupyter notebooks for data exploration, feature engineering, and model training.
- `scripts/`: Python scripts for making predictions.
- `flask_app/`: Flask app for deploying the model as an API.
- `docs/`: Additional documentation and references.
- `results/`: Evaluation results.
## How to Run the Project
1. Clone the repository from GitHub.
2. Install the dependencies: `pip install -r requirements.txt`.
3. Explore the data using `notebooks/data_exploration.ipynb`.
4. Train the model using `notebooks/model_training.ipynb`.
5. Make predictions using `scripts/predict.py`.
6. Optionally, run the Flask app with `python flask_app/app.py`.

## Requirements
- Python 3.x
- pandas
- scikit-learn
- joblib
- flask

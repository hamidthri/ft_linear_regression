# Car Price Prediction with Linear Regression

## Overview
This project implements a simple linear regression model to predict car prices based on mileage. It includes two main scripts:
- `train_model.py`: Trains the linear regression model using a dataset of car mileages and prices, saves the model parameters, and visualizes the results.
- `prediction_price.py`: Uses the trained model to predict car prices based on user-input mileage.

## Prerequisites
To run this project, you need the following Python libraries:
- `numpy`
- `matplotlib`

You can install them using pip:
```bash
pip install numpy matplotlib
```

## Dataset
The training script expects a CSV file with at least two columns: mileage and price. The first row is assumed to be a header and is skipped. Example format:
```csv
mileage,price
10000,20000
15000,18000
...
```

## Usage

### Training the Model
1. Run `train_model.py`:
   ```bash
   python train_model.py
   ```
2. Enter the path to your data file when prompted.
3. Optionally, specify the learning rate, number of iterations, and batch size (or press Enter to use defaults: learning rate=0.01, iterations=1000, batch size=32).
4. The script will:
   - Train the model using mini-batch gradient descent.
   - Display training progress and final model parameters.
   - Print precision metrics (R² score, MAE, RMSE).
   - Save the model parameters (theta0, theta1) to `model/model.txt`.
   - Generate two plots:
     - A scatter plot of the data with the fitted linear model.
     - A plot of the cost function over iterations.

### Predicting Prices
1. Run `prediction_price.py`:
   ```bash
   python prediction_price.py
   ```
2. Enter a mileage value to get the predicted price, or type `q` to quit.
3. The script will:
   - Load the model parameters from `model/model.txt` (defaults to theta0=0, theta1=0 if the file is missing or invalid).
   - Display the estimated price for the given mileage.
   - Handle invalid inputs (e.g., negative mileage or non-numeric values).

## Files
- `train_model.py`: Script for training the linear regression model.
- `prediction_price.py`: Script for predicting car prices using the trained model.
- `model/model.txt`: Stores the trained model parameters (theta0, theta1).

## Notes
- The model assumes a linear relationship between mileage and price.
- Data normalization is applied during training to improve convergence.
- The model is saved in the `model` directory, which is created automatically if it doesn't exist.
- If the data file is invalid or empty, the training script will exit with an error message.
- The prediction script defaults to zero parameters if the model file is missing or corrupted.

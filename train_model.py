import numpy as np
import matplotlib.pyplot as plt
import os


def load_data(file_path):
    mileages = []
    prices = []
    
    try:
        with open(file_path, 'r') as file:
            next(file, None)
            
            for line in file:
                line = line.strip()
                if line:
                    values = line.split(',')
                    
                    if len(values) >= 2:
                        try:
                            mileage = float(values[0])
                            price = float(values[1])
                            mileages.append(mileage)
                            prices.append(price)
                        except ValueError:
                            print(f"Skipping invalid row: {line}")
                        
    except Exception as e:
        print(f"Error reading file: {e}")
        return np.array(mileages), np.array(prices)
    
    return np.array(mileages), np.array(prices)


def normalize_data(data):
    mean = sum(data) / len(data)
    
    variance = sum((x - mean) ** 2 for x in data) / len(data)
    std = variance ** 0.5
    
    if std == 0:
        std = 1
    
    normalized_data = [(x - mean) / std for x in data]
    normalized_data = np.array(normalized_data)
    
    return normalized_data, mean, std


def denormalize_parameters(theta0, theta1, mileages_mean, mileages_std, prices_mean, prices_std):
    
    original_theta1 = theta1 * (prices_std / mileages_std)
    original_theta0 = prices_mean + prices_std * theta0 - original_theta1 * mileages_mean

    return original_theta0, original_theta1

def estimate_price(mileage, theta0, theta1):
    return theta0 + theta1 * mileage


def compute_gradients_batch(batch_mileages, batch_prices, theta0, theta1):
    m = len(batch_prices)
    tmp_theta0 = 0
    tmp_theta1 = 0
    
    for i in range(m):
        estimate = estimate_price(batch_mileages[i], theta0, theta1)
        error = estimate - batch_prices[i]
        tmp_theta0 += error
        tmp_theta1 += error * batch_mileages[i]

    return (1 / m) * tmp_theta0, (1 / m) * tmp_theta1


def compute_cost(prices, predictions):
    m = len(prices)
    cost = (1 / (2 * m)) * np.sum((predictions - prices) ** 2)
    return cost

def train_model_with_batches(mileages, prices, learning_rate=0.01, num_iterations=1000, batch_size=32):
    mileages_norm, mileages_mean, mileages_std = normalize_data(mileages)
    prices_norm, prices_mean, prices_std = normalize_data(prices)
    
    theta0 = 0
    theta1 = 0
    costs = []
    # training loop
    batch_size = min(batch_size, m)
    
    for iteration in range(num_iterations):
        for batch_start in range(0, m, batch_size):
            batch_end = min(batch_start + batch_size, m)
            batch_mileages = mileages_norm[batch_start:batch_end]
            batch_prices = prices_norm[batch_start:batch_end]
            
            grad_theta0, grad_theta1 = compute_gradients_batch(batch_mileages, batch_prices, theta0, theta1)
            
            theta0 = theta0 - learning_rate * grad_theta0
            theta1 = theta1 - learning_rate * grad_theta1
        
        if iteration % 10 == 0:
            predictions = [estimate_price(mileage, theta0, theta1) for mileage in mileages_norm]
            cost = compute_cost(prices_norm, predictions)
            costs.append(cost)
            print(f"Iteration {iteration}: Cost = {cost}, theta0 = {theta0}, theta1 = {theta1}")
            
    original_theta0, original_theta1 = denormalize_parameters(theta0, theta1, mileages_mean, mileages_std, prices_mean, prices_std)
    return original_theta0, original_theta1, costs


def main():
    data_file = input("Enter the path to the data file: ")
    
    mileages, prices = load_data(data_file)
    if len(mileages) == 0 or len(prices) == 0:
        print("No valid data found in the file.")
        return
    
    learning_rate = 0.01
    num_iterations = 1000
    batch_size = 32
    
    try:
        learning_rate = float(input("Enter the learning rate (default 0.01): ") or 0.01)
        num_iterations = int(input("Enter the number of iterations (default 1000): ") or 1000)
        batch_size = int(input(f"Enter the batch size (default {batch_size}): ") or batch_size)
    except ValueError:
        print("Invalid input. Using default values.")
        
    theta0, theta1, costs = train_model_with_batches(mileages, prices, learning_rate, num_iterations, batch_size)
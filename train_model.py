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


def calculate_precision(mileages, prices, theta0, theta1):
    
    predictions = [estimate_price(mileage, theta0, theta1) for mileage in mileages]
    
    mean_price = sum(prices) / len(prices)
    
    ss_total = sum((price - mean_price) ** 2 for price in prices)
    
    ss_residual = sum((price - pred) ** 2 for price, pred in zip(prices, predictions))
    
    # Calculate R² score
    r_squared = 1 - (ss_residual / ss_total) if ss_total != 0 else 0
    
    # Calculate Mean Absolute Error (MAE)
    mae = sum(abs(price - pred) for price, pred in zip(prices, predictions)) / len(prices)
    
    # Calculate Root Mean Squared Error (RMSE)
    rmse = (sum((price - pred) ** 2 for price, pred in zip(prices, predictions)) / len(prices)) ** 0.5
    
    return r_squared, mae, rmse

def save_model(theta0, theta1, model_dir="model"):
    try:
        os.makedirs(model_dir, exist_ok=True)
        
        model_path = os.path.join(model_dir, "model.txt")
        with open(model_path, 'w') as file:
            file.write(f"{theta0}\n{theta1}")
        
        print(f"Model saved to {model_path}")
    except Exception as e:
        print(f"Error saving model: {e}")


def plot_data_and_model(mileages, prices, theta0, theta1):

    plt.figure(figsize=(10, 6))
    plt.scatter(mileages, prices, color='blue', label='Data points')
    
    min_mileage = min(mileages)
    max_mileage = max(mileages)
    line_x = np.linspace(min_mileage, max_mileage, 100)
    line_y = [estimate_price(x, theta0, theta1) for x in line_x]
    
    plt.plot(line_x, line_y, color='red', label=f'Linear model: y = {theta0:.2f} + {theta1:.6f}x')
    
    plt.xlabel('Mileage')
    plt.ylabel('Price')
    plt.title('Car Price vs Mileage with Linear Regression')
    plt.legend()
    plt.grid(True)
    plt.show()
    
def plot_cost(costs):
    plt.figure(figsize=(10, 6))
    plt.plot(costs, color='green')
    plt.title('Cost function over iterations')
    plt.xlabel('Iterations (x100)')
    plt.ylabel('Cost')
    plt.grid(True)
    plt.show()  

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
    print(f"Training completed! Final parameters: theta0 = {theta0}, theta1 = {theta1}")
    r_squared, mae, rmse = calculate_precision(mileages, prices, theta0, theta1)
    print("\nModel Precision Metrics:")
    print(f"R² Score: {r_squared:.4f} (Higher is better, 1.0 is perfect fit)")
    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"Root Mean Squared Error: {rmse:.2f}")
    
    save_model(theta0, theta1)
    
    plot_data_and_model(mileages, prices, theta0, theta1)
    plot_cost(costs)
    
if __name__ == "__main__":
    main()
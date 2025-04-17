import os

def estimate_price(mileage, theta0, theta1):
    return theta0 + theta1 * mileage

def main():
    model_file = "model/model.txt"
    
    theta0, theta1 = 0.0, 0.0
    
    if os.path.isfile(model_file):
        try:
            with open(model_file, 'r') as file:
                line0 = file.readline().strip()
                line1 = file.readline().strip()
                
                if line0 and line1:
                    theta0 = float(line0)
                    theta1 = float(line1)
                else:
                    print("Model file is empty. Defaulting to theta0=0, theta1=0.")
        except Exception as e:
            print(f"Error reading model file: {e}")
            print("Defaulting to theta0=0, theta1=0.")
    else:
        print("Model file not found. Defaulting to theta0=0, theta1=0.")
    
    while True:
        cmd = input("Enter mileage (or 'q' to quit): ").strip().lower()
        if cmd == 'q':
            break
        
        try:
            mileage = float(cmd)
        except ValueError:
            print("Invalid input. Please enter a numeric mileage.")
            continue
        
        if mileage < 0:
            print("Mileage cannot be negative. Please enter a valid mileage.")
            continue
        
        price = estimate_price(mileage, theta0, theta1)
        print(f"Estimated price for {mileage} miles: ${price:.2f}")
    
    print("Exiting prediction tool.")

if __name__ == "__main__":
    main()

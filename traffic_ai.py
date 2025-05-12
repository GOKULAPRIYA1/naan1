# Step 1: Import necessary libraries
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Step 2: Simulated traffic data (vehicles per minute)
vehicle_count = np.array([5, 10, 15, 20, 25, 30, 35]).reshape(-1, 1)  # Input (X)
green_light_duration = np.array([10, 15, 20, 25, 30, 35, 40])        # Output (Y)

# Step 3: Train a simple linear regression model
model = LinearRegression()
model.fit(vehicle_count, green_light_duration)

# Step 4: Predict green light time for a new vehicle count
new_vehicle_count = 22  # Example real-time data
predicted_duration = model.predict([[new_vehicle_count]])

print(f"For {new_vehicle_count} vehicles/minute, set green light for {predicted_duration[0]:.2f} seconds.")

# Step 5: (Optional) Visualize
plt.scatter(vehicle_count, green_light_duration, color='blue', label='Training Data')
plt.plot(vehicle_count, model.predict(vehicle_count), color='green', label='Regression Line')
plt.scatter([new_vehicle_count], predicted_duration, color='red', label='Prediction')
plt.xlabel("Vehicle Count per Minute")
plt.ylabel("Green Light Duration (s)")
plt.title("AI Prediction for Traffic Light Timing")
plt.legend()
plt.grid(True)
plt.show()

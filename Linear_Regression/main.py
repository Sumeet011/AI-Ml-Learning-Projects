# 📊 Import required libraries
import matplotlib
matplotlib.use('Agg')  # 🔧 Use non-GUI backend (Agg) to prevent Tkinter-related errors when displaying plots in some environments (like servers)

import pandas as pd          # For handling and processing datasets
import matplotlib.pyplot as plt  # For plotting graphs

# 📥 Load dataset from CSV file
data = pd.read_csv('data.csv')

# ⚙️ Function to perform one step of gradient descent for linear regression
def gradient_descent(points, m_now, b_now, learning_rate):
    m_gradient = 0  # Gradient for slope (m)
    b_gradient = 0  # Gradient for intercept (b)
    N = len(points)  # Total number of data points
    
    # 🔁 Loop through each data point to compute gradients
    for x, y in points:
        # Partial derivative of loss w.r.t. m and b
        m_gradient += -(2/N) * x * (y - (m_now * x + b_now))
        b_gradient += -(2/N) * (y - (m_now * x + b_now))
    
    # Update parameters using the gradient and learning rate
    m = m_now - learning_rate * m_gradient
    b = b_now - learning_rate * b_gradient
    
    return m, b

# 🚀 Initialize model parameters
m = 0  # Initial slope
b = 0  # Initial intercept
learning_rate = 0.00001  # Small learning rate to ensure slow, stable updates
epochs = 1000  # Number of iterations for training

# 🔁 Perform gradient descent for a fixed number of epochs
for i in range(epochs):
    m, b = gradient_descent(data.values, m, b, learning_rate)

# 📈 Plotting the results
plt.scatter(data.x, data.y, color='blue', label='Data Points')  # Plot original data points
x_range = range(int(data.x.min()), int(data.x.max()) + 1)  # X range for the regression line
plt.plot(x_range, [m * x + b for x in x_range], color='red', label='Regression Line')  # Plot regression line
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Regression using Gradient Descent')

# 💾 Save the plot as an image (instead of showing it)
plt.savefig('regression_plot.png')
print("✅ Plot saved as 'regression_plot.png'")

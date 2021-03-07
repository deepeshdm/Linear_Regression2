import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("Salary_Data.csv")

x_values = np.array(df["YearsExperience"].values)
y_values = np.array(df["Salary"].values)


def Gradient_Descent(x, y):
    m_current = b_current = 0
    iterations = 3000
    n = len(x)
    learning_rate = 0.01

    # Calculating all the formulas at each step.
    for i in range(iterations):
        y_predicted = m_current * x + b_current
        # keep checking the cost function to measure the accuracy.
        cost_function = (1 / n) * sum([val ** 2 for val in (y - y_predicted)])
        m_derivative = -(2 / n) * sum(x * (y - y_predicted))
        b_derivative = -(2 / n) * sum((y - y_predicted))
        m_current = m_current - learning_rate * m_derivative
        b_current = b_current - learning_rate * b_derivative
        print("m : {} , b : {} , cost : {} ".format(round(m_current, 2), round(b_current, 2), round(cost_function, 2)))

    result_values = {"m": round(m_current, 2), "b": round(b_current, 2)}
    return result_values


def Predict_Salary(experience):
    result = Gradient_Descent(x_values, y_values)
    m = result.get("m")
    b = result.get("b")
    x = experience
    y = m * x + b
    print("The Predicted Salary for {} years of experience is {} ".format(x, round(y,0)))

    # plotting the dataset.
    plt.scatter(x_values, y_values, edgecolors="red", color="blue")
    plt.xlabel("Experience (years)")
    plt.ylabel("Salary")
    plt.title("Salary per Month")
    plt.show()


Predict_Salary(4)
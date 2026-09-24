import numpy as np
import matplotlib.pyplot as plt

# A1(a)
def summation_unit(x1, x2, w0, w1, w2):
    return w0 + w1 * x1 + w2 * x2

# A1(b)
def step_activation(y):
    if y >= 0:
        return 1
    return 0

# A1(c)
def comparator(target, predicted):
    return target - predicted

# A2
def train_perceptron(data, initial_weights, learning_rate, max_epochs=1000):
    w0, w1, w2 = initial_weights
    errors = []
    for epoch in range(1, max_epochs + 1):
        squared_error = 0
        for x1, x2, target in data:
            y = summation_unit(x1, x2, w0, w1, w2)
            predicted = step_activation(y)
            error = comparator(target, predicted)
            squared_error += error ** 2
            w0 = w0 + learning_rate * error
            w1 = w1 + learning_rate * error * x1
            w2 = w2 + learning_rate * error * x2
        errors.append(squared_error)
        if squared_error <= 0.002:
            return (w0, w1, w2), errors, epoch
    return (w0, w1, w2), errors, max_epochs

if __name__ == "__main__":
    and_data = [(0, 0, 0),(0, 1, 0),(1, 0, 0),(1, 1, 1)]
    xor_data = [(0, 0, 0),(0, 1, 1),(1, 0, 1),(1, 1, 0)]
    initial_weights = (-10, 0.2, -0.75)
    learning_rate = 0.05

    # A1 
    print("A1: ")
    y = summation_unit(1, 1, -10, 0.2, -0.75)
    print("Summation output:", y)
    print("Step activation:", step_activation(y))
    print("Comparator error:", comparator(1, step_activation(y)))

    # A2
    print("\nA2: AND GATE")
    weights, errors, epochs = train_perceptron(and_data, initial_weights, learning_rate)
    print("Final weights:", weights)
    print("Number of epochs:", epochs)
    print("Final error:", errors[-1])

    plt.figure()
    plt.plot(range(1, len(errors) + 1), errors, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Sum-Square Error")
    plt.title("AND Gate: Epoch vs Error")
    plt.grid(True)
    plt.show()

    # A4 
    learning_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    iterations = []
    for lr in learning_rates:
        _, _, epoch = train_perceptron(and_data, initial_weights, lr)
        iterations.append(epoch)
    print("\nA4: LEARNING RATE COMPARISON")
    for lr, epoch in zip(learning_rates, iterations):
        print("Learning rate:", lr, "Iterations:", epoch)
    plt.figure()
    plt.plot(learning_rates, iterations, marker="o")
    plt.xlabel("Learning Rate")
    plt.ylabel("Number of Iterations")
    plt.title("Learning Rate vs Number of Iterations")
    plt.grid(True)
    plt.show()

    # A5
    print("\nA5: XOR GATE")
    xor_weights, xor_errors, xor_epochs = train_perceptron(xor_data, initial_weights, learning_rate)
    print("Final weights:", xor_weights)
    print("Number of epochs:", xor_epochs)
    print("Final error:", xor_errors[-1])

    plt.figure()
    plt.plot(range(1, len(xor_errors) + 1), xor_errors)
    plt.xlabel("Epoch")
    plt.ylabel("Sum-Square Error")
    plt.title("XOR Gate: Epoch vs Error")
    plt.grid(True)
    plt.show()
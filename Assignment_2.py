import pandas as p
import numpy as n
import matplotlib.pyplot as plt
import time
#A1
def loadingdata(f):
    purchase = p.read_excel(f, sheet_name="Purchase data")
    return purchase
def matrices(m):
    A = m[["Candies (#)", "Mangoes (Kg)", "Milk Packets (#)"]].values
    y = m[["Payment (Rs)"]].values
    return A, y
def rank(A):
    r = n.linalg.matrix_rank(A)
    return r
def cost(A, y):
    inverse = n.linalg.pinv(A)
    costs = inverse @ y
    return costs
#A3
def irctc_data(f):
    stock = p.read_excel(f, sheet_name="IRCTC Stock Price")
    return stock
def price1(stock):
    price = stock["Price"].values
    return price
def numpy_mean(price):
    return n.mean(price)
def numpy_variance(price):
    return n.var(price)
def mean1(price):
    total = 0
    for value in price:
        total += value
    mean = total / len(price)
    return mean
def variance1(price):
    mean_value = mean1(price)
    total = 0
    for value in price:
        total += (value - mean_value) ** 2
    variance = total / len(price)
    return variance
def execution_time(function, data):
    total_time = 0
    for i in range(10):
        start = time.time()
        function(data)
        end = time.time()
        total_time += (end - start)
    average_time = total_time / 10
    return average_time
def price_by_day(stock, day):
    data = stock[stock["Day"] == day]
    return data["Price"].values
def price_by_month(stock, month):
    data = stock[stock["Month"] == month]
    return data["Price"].values
def loss_prob(stock):
    changes = stock["Chg%"].values
    loss = list(filter(lambda x: x < 0, changes))
    prob = len(loss) / len(changes)
    return prob
def profit_wednesday_probability(stock):
    profitable_wednesday = stock[
        (stock["Day"] == "Wed")
        & (stock["Chg%"] > 0)]
    probability = len(profitable_wednesday) / len(stock)
    return probability
def conditional_probability(stock):
    profitable_wednesday = stock[
        (stock["Day"] == "Wed")
        & (stock["Chg%"] > 0)]
    total_wednesday = stock[
        stock["Day"] == "Wed"]
    probability = len(profitable_wednesday) / len(total_wednesday)
    return probability
def scatter_plot(stock):
    plt.scatter(stock["Day"], stock["Chg%"])
    plt.xlabel("Day")
    plt.ylabel("Chg%")
    plt.title("Change Percentage vs Day")
    plt.show()
#Main function
def main():
    f = "Lab Session Data.xlsx"
    #A1
    purchase = loadingdata(f)
    A, y = matrices(purchase)
    r = rank(A)
    costs = cost(A, y)
    print("Feature Matrix X:\n", A)
    print("Output Vector y:\n", y)
    print("Rank of Matrix X =", r)
    print("Product Costs:")
    print("Candy =", costs[0][0])
    print("Mango =", costs[1][0])
    print("Milk Packet =", costs[2][0])
    print()
    #A3
    stock = irctc_data(f)
    price = price1(stock)
    numpy_mean_value = numpy_mean(price)
    numpy_variance_value = numpy_variance(price)
    mean_value = mean1(price)
    variance_value = variance1(price)
    mean_time = execution_time(mean1, price)
    variance_time = execution_time(variance1, price)
    numpy_mean_time = execution_time(numpy_mean, price)
    numpy_variance_time = execution_time(numpy_variance, price)
    wednesday_price = price_by_day(stock, "Wed")
    wednesday_mean = numpy_mean(wednesday_price)
    april_price = price_by_month(stock, "Apr")
    april_mean = numpy_mean(april_price)
    loss = loss_prob(stock)
    profit_wednesday = profit_wednesday_probability(stock)
    conditional_profit = conditional_probability(stock)
    print("Mean NumPy =", numpy_mean_value)
    print("Variance NumPy =", numpy_variance_value)
    print("Mean =", mean_value)
    print("Variance =", variance_value)
    print("Execution Time Custom Mean =", mean_time)
    print("Execution Time Custom Variance =", variance_time)
    print("Execution Time NumPy Mean =", numpy_mean_time)
    print("Execution Time NumPy Variance =", numpy_variance_time)
    print("Population Mean =", numpy_mean_value)
    print("Wednesday Mean =", wednesday_mean)
    print("April Mean =", april_mean)
    print("Probability of Loss =", loss)
    print("Probability of Profit on Wednesday =", profit_wednesday)
    print("Conditional Probability of Profit given Wednesday =",conditional_profit)
    scatter_plot(stock)

if __name__ == "__main__":
    main()
import random
import statistics

# Q1
def count(a):
    vowels = "aeiouAEIOU"
    vowelscount = 0
    consonantcount = 0
    for letters in a:
        if letters in vowels:
            vowelscount = vowelscount + 1
        elif letters.isalpha():
            consonantcount = consonantcount + 1
    return vowelscount, consonantcount

# Q2
def multiply(m1, c1, m2, c2):
    if c1 != m2:
        return None
    A = [[0] * c1 for i in range(m1)]
    for i in range(m1):
        for j in range(c1):
            A[i][j] = int(input())
    B = [[0] * c2 for i in range(m2)]
    for i in range(m2):
        for j in range(c2):
            B[i][j] = int(input())
    result = [[0] * c2 for i in range(m1)]
    for i in range(m1):
        for j in range(c2):
            total = 0
            for k in range(c1):
                total = total + A[i][k] * B[k][j]
            result[i][j] = total
    return result

# Q3
def common(a, b):
    c = 0
    for i in range(a):
        for j in range(b):
            if l1[i] == l2[j]:
                c += 1
                break
    return c

# Q4
def transpose(r, c):
    T = [[0] * r for i in range(c)]
    for i in range(c):
        for j in range(r):
            T[i][j] = A[j][i]
    return T

# Q5
def randomnumbers():
    a = []
    for i in range(100):
        a.append(random.randint(100, 150))
    return a, statistics.mean(a), statistics.median(a), statistics.mode(a)

# Q1
a = input("Enter a string: ")
vowelscount, consonantcount = count(a)
print("No. of vowels are:", vowelscount)
print("No. of consonants are:", consonantcount)
# Q2
m1 = int(input("Enter rows of A: "))
c1 = int(input("Enter columns of A: "))
m2 = int(input("Enter rows of B: "))
c2 = int(input("Enter columns of B: "))
result = multiply(m1, c1, m2, c2)
if result == None:
    print("Error: U can't multiply.")
else:
    print("Product Matrix:")
    print(result)
# Q3
a = int(input("Enter size of 1st list: "))
l1 = []
print("Enter elements of 1st list:")
for i in range(a):
    l1.append(int(input()))
b = int(input("Enter size of 2nd list: "))
l2 = []
print("Enter elements of 2nd list:")
for i in range(b):
    l2.append(int(input()))
c = common(a, b)
print("No. of common elements:", c)
# Q4
r = int(input("Enter rows: "))
c = int(input("Enter columns: "))
A = [[0] * c for i in range(r)]
print("Enter matrix:")
for i in range(r):
    for j in range(c):
        A[i][j] = int(input())
T = transpose(r, c)
print(T)
# Q5
a, mean, median, mode = randomnumbers()
print(a)
print("Mean:", mean)
print("Median:", median)
print("Mode:", mode)
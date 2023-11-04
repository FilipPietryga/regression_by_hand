import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('data.csv')

plt.scatter(data.studytime, data.score)
plt.show()

def loss(a, b, data):
  error = 0
  for i in range(len(data)):
    x = data.iloc[i].studytime
    y = data.iloc[i].score
    error += (y - (a*x+b)) ** 2
  error /= float(len(data))
  
def gradient(a,b, data, L):
  a_grad = 0
  b_grad = 0
  n = len(data)
  
  for i in range(n):
    x = data.iloc[i].studytime
    y = data.iloc[i].score
    
    a_grad += -2/n * x * (y - (a*x+b))
    b_grad += -2/n * (y - (a*x+b))
  
  a_new = a - a_grad * L
  b_new = a - a_grad * L
  
  return a_new, b_new

a = 0
b = 0
L = 0.001
epochs = 1000

for i in range(epochs):
  print("Epoch:" + str(i))
  a,b = gradient(a,b,data,L)
  
print(a,b)

plt.scatter(data.studytime, data.score, color="black")
plt.plot(list(range(0, 20)), [a*x+b for x in range(0, 20)], color="red")
plt.show()

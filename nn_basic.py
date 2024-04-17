import numpy as np 
import matplotlib.pyplot as plt

x = np.array([1,2,3,4,5])
y = np.array([2,5,8,10,13])


class lrmodel():
    def __init__(self):
        self.w = 0.9
        self.b = 0.3
    def forward(self, x):
        return self.w * x + self.b
    
m = lrmodel()

lr = 0.01
num_epochs = 50
history = []

for i in range(num_epochs+1):
    history.append(np.sum(np.square(m.forward(x) - y)))
    if i%5 == 0:
        print(f"Error {i}: {np.sum(np.square(m.forward(x) - y))}")
    m.w -= lr/5 * (np.sum(x * (m.forward(x) - y)))
    m.b -= lr/5 * (np.sum(m.forward(x) - y))
    
print(f"The final equation is: y = {m.w} * x + {m.b}")

plt.xlabel("num_epochs")
plt.ylabel("Mse Loss")
plt.plot(history)
plt.show()
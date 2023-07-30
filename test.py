import numpy as np
import matplotlib.pyplot as plt


'''
(a) A Rule-Based Classifier, 𝑅, has the following rules: 
if 𝑥1 > 4 then class 2
else if 𝑥2 < 6 then class 1
else class 3
Sketch the classification regions for a feature space defined as 𝑥1 ∈ 1..9 and 𝑥2 ∈ −2..8.
'''


# Create a grid of points in the feature space

x1 = [1,2,3,4,5,6,7,8,9]
x2 = [-2,-1,0,1,2,3,4,5,6,7,8]
X1, X2 = np.meshgrid(x1, x2)

class_labels = np.zeros_like(X1)
class_labels[X1 > 4] = 2
class_labels[X2 < 6] = 1
class_labels[(X1 > 4) & (X2 < 6)] = 3

# Plot the classification regions
plt.figure()
plt.contourf(X1, X2, class_labels, levels=[0, 1, 2, 3], colors=['red', 'blue', 'yellow'])
#plt.pcolormesh(X1, X2, class_labels, shading='auto')
plt.colorbar(label='Class Labels')
plt.xlabel('x1')
plt.ylabel('x2')
plt.text(2, 3, 'Class 1', color='black', ha='center', va='center')
plt.text(6, 2, 'Class 2', color='black', ha='center', va='center')
plt.text(2, 7, 'Class 3', color='black', ha='center', va='center')


plt.title('Classification Regions')
plt.grid(True)
plt.show()

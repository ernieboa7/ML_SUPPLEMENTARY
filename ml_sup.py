import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import tools_4211 as ts
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from tools_4211 import train_test_knn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix

'''
1. A 1-dimensional data-set is shown in Figure 1. There are two classes; class 1/negative is
shown with black circles, class 2/positive are shown with red triangles.

(a) Use a Binary Classifier with this data-set, calculate the confusion matrices for each [10]
threshold.
(b) Use the confusion matrices from the previous part to produce the corresponding ROC [5]
Curve.
(c) Calculate the Operational Point for this classifier. Show all of your working. [5]
'''
X_axis=[]
Sensitivity=[]
def Confusion_matrix(TP, FP, FN, TN):
    Sensitivity.append(TP/(TP+FN+1e-8))
    Specificity = TN/(TN+FP+1e-8)
    X_axis.append(1-Specificity)
    #print('Sensitivity=', Sensitivity)
    #print('Specificity=', Specificity)
    #print('X_axis=', X_axis)
    return Sensitivity, Specificity, X_axis
    



# FIRST THRESHOLD
TP=4; FN=4; FP=0; TN=0
Confusion_matrix(TP,FP,FN,TN)
# SECOND THRESHOLD
TP=3; FN=4; FP=1; TN=0
Confusion_matrix(TP,FP,FN,TN)
# THIRD THRESHOLD
TP=2; FN=4; FP=2; TN=0
Confusion_matrix(TP,FP,FN,TN)
# FORTH THRESHOLD
TP=2; FN=3; FP=2; TN=1
Confusion_matrix(TP,FP,FN,TN)
# FIFTH THRESHOLD
TP=2; FN=2; FP=2; TN=2
Confusion_matrix(TP,FP,FN,TN)
# SIXTH THRESHOLD
TP=2; FN=1; FP=2; TN=3
Confusion_matrix(TP,FP,FN,TN)
# SEVENTH THRESHOLD
TP=1; FN=1; FP=3; TN=3
Confusion_matrix(TP,FP,FN,TN)
# EIGHT THRESHOLD
TP=1; FN=0; FP=3; TN=4
Confusion_matrix(TP,FP,FN,TN)
# NINTH THRESHOLD
TP=0; FN=0; FP=4; TN=4
Confusion_matrix(TP,FP,FN,TN)

print('Sensitivity=', Sensitivity)
#print('Specificity=', Specificity)
print('X_axis=', X_axis)

plt.figure()
plt.grid()
plt.plot(X_axis, Sensitivity, color="orange", marker="o" ) #'c.-'
plt.xlabel('X-axis')
plt.ylabel('Sensitivity')
plt.title('ROC Curve')
plt.plot([0.42857142938775505], [0.9999999900000002], color="green", marker="*")
   
plt.show() 
   

from numpy import sqrt

Z1 = sqrt((0.42857142938775505**2) + ((1-0.9999999900000002)**2))
print('Z1=',Z1)
Z2 = sqrt((0.4000000012**2)+((1-0.6666666644444444)**2))
print('Z2=',Z2)

# Z1= 0.42857142938775517, Z2=Z2= 0.5206833140716078:
# There Z1 is the operational point






'''
2. Figure 2 shows a 2-dimensional data-set with 3 classes. There are two new data points at
(0.2, 0.6) and (0.9, 0.2) shown with orange triangles.
(a) Using the k-NN Classifier, suggest class labels for the new data points where:
i. k = 1 [1]
ii. k = 4 [1]
(b) Train a Nearest-Mean-Classifier and classify the new data points indicated with orange [11]
triangles. Show all calculations.
(c) Suggest a single Linear Discriminant Classifier for this data-set to minimise the error [7]
rate. Provide the equation and the confusion matrix of the classifier. Include all
working.'''

training_data = np.array([[[0.3, 0.3], [0.3, 0.5], [0.6, 0.1], [0.7, 0.8]],
                        [[0.2, 0.2], [0.4, 0.3], [0.7, 0.1], [0.0, 0.0]],
                        [[0.6, 0.2], [0.8, 1.0], [0.9, 0.7], [0.0, 0.0]]])

print(training_data)


testing_data = np.array([[0.1, 0.4], [0.3, 0.5]])
testing_labels = np.array(['red', 'green', 'blue'])

X_train = training_data.reshape(-1, 2)
y_train = np.repeat(testing_labels, 4)

X_test = testing_data

def train_test_knn(X_train, y_train, testing_data,testing_labels, k = 1):
    """
    Trains and tests a decision tree classifier.
    Returns the classification error, the predicted labels and the classifier.
    """
    cla = KNeighborsClassifier(n_neighbors = k)
    # Train the classifier with data and labels
    cla.fit(X_train, y_train)
    assigned_labels = cla.predict(testing_data)
    #testing_error = np.mean(testing_labels != assigned_labels)
    return assigned_labels, cla

knn1 =train_test_knn(X_train, y_train, testing_data,testing_labels, k = 1) 
print(knn1)  

knn4 =train_test_knn(X_train, y_train, testing_data,testing_labels, k = 4) 
print(knn4) 



# 2b
class_labels = ['red', 'green', 'blue']

# Mean vectors for each class
mean_vectors = np.mean(training_data, axis=1)

# Test data
test_data = np.array([[0.1, 0.4], [0.3, 0.5]])


# Calculate distances and classify
for i in range(len(test_data)):
    distances = np.linalg.norm(test_data[i] - mean_vectors, axis=1)
    min_index = np.argmin(distances)
    classification = class_labels[min_index]
    print(f"Nearest-Mean-Classifier new data point {i+1}: Class = {classification}")
    
    
#2c
test_data = np.array([[0.1, 0.4], [0.3, 0.5], [0.0, 0.0]])

error, predicted_labels, classifier = ts.train_test_ldc(X_train, y_train, X_train, y_train)
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
assigned_labels = lda.predict(test_data)
error_rate = np.mean(testing_labels != assigned_labels)
print('assigned_label', assigned_labels)
print('error_rate', error_rate)



# Compute the confusion matrix
confusion_mat = confusion_matrix(y_train, predicted_labels)

print("Confusion Matrix:")
print(confusion_mat)




'''
3. Figure 3 shows a time-series of attendees by day to a two-week event.
Calculate (show all working):
(a) the mean attendance over the whole event. [3]
(b) the median value. [3]
(c) the number of days that attendance will be higher than predicted by the linear [14]
regression equation 𝑦 = −0.037𝑥 + 10.923, where x is the day of the event (1...14).
Show all working.'''


# Solution
Attendees = [18, 17, 17, 6, 14, 11, 17, 18, 11, 14, 6, 17, 8, 6]
Sorted_Attendees = sorted(Attendees)
print('Sorted_Attendees: ', Sorted_Attendees)

Days = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
No_of_days = 14

#(a)
mean_attendance = sum(Sorted_Attendees)/No_of_days
print('mean_attendance: ', mean_attendance)

#3(b)
median_attendance_index = ((No_of_days/2 + 1) +  (No_of_days/2)) / 2
print('median_attendance_index: ', median_attendance_index)
# The median index number is 7.5 i.e between 7 and 8
Sorted_Attendees = [6, 6, 6, 8, 11, 11, 14, 14, 17, 17, 17, 17, 18, 18]
# the index number of 7 and 8 in the sorted array is 14 and 14. Therefore the median is (14+14)/2 = 14
# Using numpy to double check the median
a = np.median(Sorted_Attendees)
print(a)


#3c
y = lambda x: -0.037 * x + 10.923
num_days_with_higher_attendance = sum(actual > y(Days) for Days, actual in enumerate(Sorted_Attendees, start=1))
print("Number of days with higher attendance than predicted:", num_days_with_higher_attendance)


'''
4. (a) A Rule-Based Classifier, 𝑅, has the following rules: [5]
if 𝑥1 > 4 then class 2
else if 𝑥2 < 6 then class 1
else class 3
Sketch the classification regions for a feature space defined as 𝑥1 ∈ 1..9 and 𝑥2 ∈ −2..8.
(b) Calculate the accuracy for classifier 𝑅, when testing with the following points: [3]
x1 x2 Label
1 8 1
3 5 3
0 8 3
6 9 2
4 4 2
(c) Is the testing data-set from part b) balanced or imbalanced? Explain your answer. [2]
(d) Using the following data-set 𝑍:
x1 x2 Label
9 0 0
1 7 0
1 5 1
8 2 2
5 1 0
9 7 2
8 -1 2
5 -2 1
4 2 1
4 8 1
i. Calculate the prior probabilities of each class. [4]
ii. Which class would the ZeroR classifier assign? [1]
iii. Assume that class 2 is the positive class, and the other two classes are pooled to [5]
make the negative class.
Calculate the confusion matrix for this case and subsequently calculate the
specificity and F1 measure for classifier 𝑅 with data-set 𝑍.''''''


(a) A Rule-Based Classifier, 𝑅, has the following rules: 
if 𝑥1 > 4 then class 2
else if 𝑥2 < 6 then class 1
else class 3
Sketch the classification regions for a feature space defined as 𝑥1 ∈ 1..9 and 𝑥2 ∈ −2..8.
'''


x1 = [1,2,3,4,5,6,7,8,9]
x2 = [-2,-1,0,1,2,3,4,5,6,7,8]
X1, X2 = np.meshgrid(x1, x2)

class_labels = np.zeros_like(X1)

class_labels[X1 > 4] = 2
class_labels[X2 < 6] = 1
class_labels[(X1 <= 4) & (X2 >= 6)] = 3

# Plot the classification regions
plt.figure()
plt.pcolormesh(X1, X2, class_labels, shading='auto')

# Label different regions
plt.text(2, 3, 'Class 1')
plt.text(6, 2, 'Class 2')
plt.text(2, 7, 'Class 3')

plt.colorbar(label='Class Labels')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Classification Regions')
plt.grid(True)
plt.show()

'''(b) Calculate the accuracy for classifier 𝑅, when testing with the following points: [3]
x1 x2 Label      
1  8  1          
3  5  3          
0  8  3          
6  9  2          
4  4  2          

Answer 
Using the classification region and the statement in 4a
if 𝑥1 > 4 then class 2
else if 𝑥2 < 6 then class 1
else class 3


x1 x2 Label      Prediction
1  8  1          False
3  5  3          False
0  8  3          True
6  9  2          True
4  4  2          True

The prediction was right 3 out of 5 therefore the accuracy is 3/5= 0.6

(c) Is the testing data-set from part b) balanced or imbalanced? Explain your answer.
Answer:
The data_set is inbalanced because the instances at whch it predicted the labels are not the same
for example it has predicted label 1 wrongly and predicted label 2 correctly twice.'''


'''
(d) Using the following data-set 𝑍:
x1 x2 Label
9  0  0
1  7  0
1  5  1
8  2  2
5  1  0
9  7  2
8 -1  2
5 -2  1
4  2  1
4  8  1
i. Calculate the prior probabilities of each class. [4]
Solution:
Prior probability for class label_0
3/10 = 0.3

Prior probability for class label_1
4/10 = 0.4

Prior probability for class label_2
3/10 = 0.3


ii. Which class would the ZeroR classifier assign? [1]
Solution:
The largest class is class_label_1 which is 4 out of 10. Therefore the ZeroR classifier
would be assign to class_label_1

iii. Assume that class 2 is the positive class, and the other two classes are pooled to [5]
make the negative class.
Calculate the confusion matrix for this case and subsequently calculate the
specificity and F1 measure for classifier 𝑅 with data-set 𝑍.

Solution:
Confusion matrix
              Assign
          TP     FN
Actual    FP     TN

'''
TP=3; FN =3; FP=0; TN=4
Specificity = TN / (TN + FP) 
print('specificity =',Specificity)

#The F1 measure is the harmonic mean of precision and recall. It is calculated as:
Precision = TP / (TP + FP) 
print('precision = ', Precision)

Recall = TP / (TP + FN)
print('recall= ', Recall) 

F1 = 2 * (Precision * Recall) / (Precision + Recall)
print('F1 measure = ', F1)





'''
5. (a) What are the four types of data analytics? [4]
(b) What type of distribution is shown in Figure 4? [1]
(c) Sketch a Box and Whisker plot for the following data-set: [5]
12, 9, 13, 10, 8, 16, 10
(d) What is the formula for calculating the Standard Deviation of a population? [2]
(e) No matter what type of Regression is used, what is the goal of the algorithm? [2]
(f) What three visualisation methods would be appropriate for 2-dimensional continuous- [3]
valued function?
(g) Sketch an appropriate visualisation of the following data: [3]
Month Rainfall (mm)
August 2022 100
September 2022 62
October 2022 111'''

'''
(a) The four types of data analytics are:

Descriptive Analytics 
Diagnostic Analytics 
Predictive Analytics 
Prescriptive Analytics 

(b) The distribution in figure 4 is a histogram.


(c) Sketch a Box and Whisker plot for the following data-set: [5]
12, 9, 13, 10, 8, 16, 10'''


data = [12, 9, 13, 10, 8, 16, 10]
sorted_data = sorted(data)

plt.boxplot(sorted_data)
plt.xlabel('data-set')
plt.ylabel('Values')
plt.title('Box and Whisker Plot in Question 5c')
plt.show()

'''
(d) The formula for calculating the standard deviation of a population is:

σ = sqrt(Σ(x - μ)² / N)

Where:

σ = population standard deviation.
Σ = summation symbol.
x = individual value in the population.
μ = population mean.
N = total number of values in the population.

(e) The role of regression algorithms is to find the relationship between independent variables (inputs) and dependent variables (output) in order to make predictions or estimate values.

(f) Three visualization methods appropriate for a 2-dimensional continuous-valued function are:

Contour Plot 
Scatter plot: 
Heatmap: 

'''
'''(g) Sketch an appropriate visualisation of the following data: [3]
Month Rainfall (mm)
August 2022 100
September 2022 62
October 2022 111'''
months = ['August 2022', 'September 2022', 'October 2022']
rainfall = [100, 62, 111]

plt.bar(months, rainfall)
plt.xlabel('Months')
plt.ylabel('Rainfall (mm)')
plt.title('Rainfall Data: Question 5g')
plt.show()
#The bar chart will display the rainfall (in mm) for each month, with the x-axis representing the months and 
#the y-axis representing the rainfall values.







# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load necessary libraries for data handling, metrics, and visualization.

2.Read the dataset using pd.read_csv() and display basic information.

3.Set initial values for slope (m), intercept (c), learning rate, and epochs.

4.Perform iterations to update m and c using gradient descent.

5.Visualize the error over iterations to monitor convergence of the model.

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: Thaksha Rishi
RegisterNumber:  212223100058
*/
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1,y,learning_rate=0.01,num_iters=1000):
    X=np.c_[np.ones(len(X1)),X1]
    theta=np.zeros(X.shape[1]).reshape(-1,1)
    for _ in range(num_iters):
        predictions=(X).dot(theta).reshape(-1,1)
        errors=(predictions-y).reshape(-1,1)
        theta_=learning_rate*(1/len(X1))*X.T.dot(errors)
        pass
    return theta
data=pd.read_csv('50_Startups.csv',header=None)
print(data.head())

X=(data.iloc[1:, :-2].values)
print(X)

X1=X.astype(float)
scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
print(y)

X1_Scaled=scaler.fit_transform(X1)
Y1_Scaled=scaler.fit_transform(y)
print(X1_Scaled)
print(Y1_Scaled)

theta=linear_regression(X1_Scaled,Y1_Scaled)
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_Scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(f"Predicted value: {pre}")
```

## Output:

<img width="644" height="156" alt="image" src="https://github.com/user-attachments/assets/ac2c5334-76fe-41b6-9a7f-dd2a600b2bbf" />

<img width="349" height="491" alt="image" src="https://github.com/user-attachments/assets/c1e5b1ab-aeb2-49ed-86c2-6037a4317dfc" />

<img width="324" height="480" alt="image" src="https://github.com/user-attachments/assets/d403ed5f-0d0e-4aee-9939-a95b3d6e5d16" />

<img width="150" height="546" alt="image" src="https://github.com/user-attachments/assets/663251f5-98fe-4bdb-b9af-7545fbcbb2ea" />


<img width="117" height="425" alt="image" src="https://github.com/user-attachments/assets/a2d28b48-fa38-4bfe-998b-b6dd4bbc1a5c" />


<img width="460" height="600" alt="image" src="https://github.com/user-attachments/assets/a0b07f92-610f-4c8c-94a3-cb3eeaa5b0b5" />


<img width="442" height="582" alt="image" src="https://github.com/user-attachments/assets/e2983df6-ff57-4b4d-a40d-968db16fe7e8" />


<img width="327" height="699" alt="image" src="https://github.com/user-attachments/assets/e744f846-d3b5-497b-aa5f-addd700f260e" />

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.

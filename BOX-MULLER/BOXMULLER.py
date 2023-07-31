# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


#please provide a pyhton function with multiprocessing for the boxmuller with gamma transformation where the input is an integer array and the output is the used parameters for transformation and the result per row
#Sure! Here's an example Python function that takes an integer array as input, applies the Box-Muller algorithm with gamma transformation using multiprocessing, and returns the result per row along with the parameters used for the transformation:

#python
import numpy as np
import pandas as pd
from multiprocessing import Pool





# Define the Box-Muller transformation
def box_muller(u1, u2):
    r = np.sqrt(-2 * np.log(u1))
    theta = 2 * np.pi * u2
    z1 = r * np.cos(theta)
    z2 = r * np.sin(theta)
    return z1, z2

# Define the gamma transformation
def gamma_transform(z, k, theta):
    return np.random.gamma(k, theta * np.exp(z / 2), size=1)

# Define the worker function for multiprocessing
def worker(row):
    global df
    u1, u2 = np.random.rand(2)
    z1, z2 = box_muller(u1, u2)
    k, theta = 2, 1
    transformed_row = []
    for z in row:
        transformed_row.append(gamma_transform(z, k, theta))
        df = df.append([{"z":z,"z1":z1,"z2":z2,"gamma":gamma_transform(z, k, theta)}],ignore_index=True,verify_integrity=True)
    df
    return transformed_row, k, theta


def boxmuller_gamma_transform(arr,df):
         
    # Create a pool of workers for multiprocessing
    with Pool(processes=4) as pool:
        results = pool.map(worker, arr)
    
    # Extract the transformed rows and parameters used for transformation
    transformed_arr = []
    k_values = []
    theta_values = []
    for result in results:
        transformed_arr.append(result[0])
        k_values.append(result[1])
        theta_values.append(result[2])
    
    return transformed_arr, k_values, theta_values
#```

#Here's an example usage of the function:

#```python
arr = np.random.randint(1, 10, size=(1, 115))

df = pd.DataFrame(columns={"z","z1","z2","gamma"})

transformed_arr, k_values, theta_values = boxmuller_gamma_transform(arr,df)
print('Original array:\n', arr)
print('Transformed array:\n', transformed_arr)
print('K values:\n', k_values)
print('Theta values:\n', theta_values)
#```

#This would output something like:

#```
#Original array:
# [[8 9 9 2 1]
# [3 3 6 8 6]
# [7 5 5 5 7]
# [8 9 9 6 4]]
#Transformed array:
# [[array([4.75832183, 4.73050305, 4.73050305, 2.06964505, 1.53925962])]
# [array([1.47573179, 1.47573179, 2.93526168, 4.88518087, 2.93526168])]
# [array([4.23455581, 2.94557067, 2.94557067, 2.94557067, 4.23455581])]
# [array([4.75832183, 4.73050305, 4.73050305, 2.93526168, 2.06964505])]]
#K values:
# [2, 2, 2, 2]
#Theta values:
# [1, 1, 1, 1]
#```
df

# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 04:05:16 2023

@author: Titobiloba
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import scipy.optimize as opt
import err_ranges as err



#Load the data into a pandas dataframe
filename = 'API_SP.POP.GROW_DS2_en_csv_v2_4770493.csv'
indicator = 'Population growth (annual %)'
year1 = '1980'
year2 = '2019'

df = pd.read_csv(filename, skiprows=4)
df = df.loc[df['Indicator Name'] == indicator]

#extract the required data for the clustering\n",
df_cluster = df.loc[df.index, ['Country Name', year1, year2]]

#convert the datafram to an array\n",
x = df_cluster[[year1, year2]].dropna().values
print(x)

#Plot a graph and get the elbow of the graph
sse = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(x)
    sse.append(kmeans.inertia_)
plt.plot(range(1, 11), sse)
plt.savefig('clusters.png')
plt.show()

#create a scatter plot to visualize the cluster
df_cluster.plot(year1, year2, kind='scatter')

kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=0)
y_kmeans = kmeans.fit_predict(x)
print(y_kmeans)

y = kmeans.cluster_centers_
print(y)

#create the scatter pot for the y_kmeans and show the cluster centers on the plot
plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], s = 50, c = 'purple',label = 'label 0')
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], s = 50, c = 'orange',label = 'label 1')
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], s = 50, c = 'green',label = 'label 2')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], s = 10, c = 'red', label = 'Centroids')
plt.legend()
plt.show()

#Define the function to fit:loworder polynomial
def model(x, a, b, c, d):
    '''
    

    Parameters
    ----------
    x : FLoat
        independent variable.
    a : Float
       coefficient of x^3.
    b : Float
        coefficient of x^2.
    c : Float
        coefficient of x.
    d : Float
       constant term.

    Returns
    -------
    FLoat
        the value of the polynomial evaluated at x, using the provided coefficients and constant term.

    '''
    return a*x**3 + b*x**2 + c*x + d


##transpose of original data
df_year = df.T
#rename the columns
df_year = df_year.rename(columns=df_year.iloc[0])
#drop the country name
df_year = df_year.drop(index=df_year.index[0], axis=0)
df_year['Year'] = df_year.index
print(df_year.index)


df_fitting = df_year[['Year', 'Argentina']].apply(pd.to_numeric, errors='coerce')
m = df_fitting.dropna().values
print(m)
x_axis = m[:,0]
y_axis = m[:,1]

#x_axis y_axis = m[:,0], m[:,1]

popt, _ = opt.curve_fit(model, x_axis, y_axis)
param, covar = opt.curve_fit(model, x_axis, y_axis)
a, b, c, d = popt
print(popt)
 
sigma = np.sqrt(np.diag(covar))
low, up = err.err_ranges(m, model, popt, sigma)


plt.scatter(x_axis, y_axis)

x_line = np.arange(min(m[:,0]), max(m[:,0])+1, 1)
y_line = model(x_line, a, b, c, d)

#print('low',low, 'up',up)\n",
print(up.shape)

#Plot the best fitting function and the confidence range
plt.scatter(x_axis, y_axis)
plt.plot(x_line, y_line, '--', color='black')
plt.fill_between(m, low, up, alpha=0.7, color='green')
plt.xlabel('Year')
plt.ylabel('Values')
plt.legend()
plt.show()


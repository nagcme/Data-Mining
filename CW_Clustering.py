'''
References: https://scikit-learn.org/stable/modules/clustering.html#k-means
            https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.euclidean_distances.html
Assumptions:
            1. The input dataset 'wholesale_customers.csv' is present in the same directory as the python file
'''

# Import Statements
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.cluster as cluster
import sklearn.metrics as metrics
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

# Read the dataset wholesale_customers.csv into a dataframe
customer = pd.read_csv('wholesale_customers.csv',
                    usecols = ['Fresh','Milk','Grocery','Frozen','Detergents_Paper','Delicassen'],
                    header=0)

# Question 1
# Create a table in the report with the mean µj = Pm i=1 xi,j and range [xj,min, xj,max]
# for each attribute j, where xi,j is the attribute j value of instance i and xmin,j , xmax,j are the
# minimum and maximum attribute j values among all instances.
print('***********************************************************')
print('Clustering Answer 1:')
print('***********************************************************')

# Create a list to print the mean and range of all the attributes
column_names = ["Attribute", "Mean", "Min", "Max"]

attr_df = pd.DataFrame(columns = column_names)

# Create the list of values for each attribute of the dataset
for col in customer.columns:
    df_list = [col,round(customer[col].mean(),2),min(customer[col]),max(customer[col])]
    df_length = len(attr_df)
    attr_df.loc[df_length] = df_list
# Print the list of mean and range of each attributes
print(attr_df)

# Checking outliers using describe()
#print(customer.describe())

# Question 2
# Run k-means with k = 3 and construct a scatterplot for each pair of attributes using
# Pyplot. Therefore, 15 scatter plots should be constructed in total. Different clusters should
# appear with different colors in the scatter plot. All scatter plots should be included in the
# report, using no more than two pages for them

print('***********************************************************')
print('Clustering Answer 2:')
print('***********************************************************')

K = 3
count1=0
count2=0
# Construct the K-Means Clustering to fit the customer dataset
# random state is taken as 0 so that the same set of result is
# obtained during each run
km = cluster.KMeans(n_clusters=K, random_state=0)
X = customer
X = X.to_numpy()

# The K-means clustering model is then fitted to the customer dataset
km.fit(X)

# As there are 6 attributes, pairwise attributes are considered to
# graphically represent the fit of the model
fig, ax = plt.subplots(5, 3, figsize = (20, 20), constrained_layout = True)
for i in range(len(customer.columns)):
    for j in range(i+1,len(customer.columns)):
        # fetch each pair of attributes to plot in the scatter plot
        X = customer.iloc[:,[i,j]]
        col1 = X.columns[0]
        col2 = X.columns[1]
        X = X.to_numpy()

        # plot the pairwise attributes in the subplot
        ax[count2,count1].scatter(X[:,0], X[:,1], c=km.labels_, cmap='viridis',
                                  marker = '.', s = 10)
        ax[count2,count1].set_title(col1+' vs '+col2)
        ax[count2,count1].set_xlabel(col1)
        ax[count2,count1].set_ylabel(col2)

        count1 = count1 + 1
        if count1 == 3:
            count2 = count2 + 1
            count1 = 0
plt.show()

# Question 3
# Run k-means for each possible value of k in the set {3, 5, 10}. Complete the following
# table with the between cluster distance BC, within cluster distance W C and ratio BC/W C of
# the set of clusters obtained for each k

print('***********************************************************')
print('Clustering Answer 3:')
print('***********************************************************')

# create a list of the possible values of K
K = [3,5,10]

# for each value of K, run the k-means and calculate the ratio of BC/WC
for k in K:
    dist_sum = 0.0
    X = customer.to_numpy()
    # run k-means for each value of k
    km = cluster.KMeans(n_clusters=k, random_state=0)
    # fit the dataset in the model
    km.fit(X)

    #Calculation of BC/WC ratio

    # Within Cluster distance is calculated using the inertia function
    WC = km.inertia_

    # Euclidean distance between all the clusters are calculated
    dist_clusters = euclidean_distances(km.cluster_centers_)
    l = 0
    # Calculation of between cluster distance
    for d in dist_clusters:
        for dis in range(l,len(d)):
            dist_sum = dist_sum + pow(d[dis],2)
        l = l+1
    BC = dist_sum
    # print the BC, WC and the ratio for each value of K
    print('K = '+str(k))
    print('BC = '+str(round(BC,2)))
    print('WC = '+str(round(WC,2)))
    print('BC/WC = '+str(round((BC/WC),2)))

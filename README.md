# Data-Mining
Clustering and Classification

Versions:
python '3.8.5'
pandas '1.1.4'
sklearn '0.24.1'
matplotlib '3.3.2'
numpy '1.19.3'

**********************************************************************************************
****************************************CLASSIFICATION****************************************
**********************************************************************************************

Files:
CW_Classification.py
adult.csv

Code Structure:

1. The input dataset 'adult.csv' is present in the same directory as the python file
2. Read adult.csv, only the below columns are imported in dataframe:
	['age','workclass','education','education-num','marital-status','occupation',
	 'relationship','race','sex','capitalgain','capitalloss','hoursperweek','native-country']
3. Calculate the below characteristics by using pandas built in function
	No. of instances : Total no. of records in the dataset
	No. of missing values : Total no. of missing values
	Fraction of missing values over all attributes: No. of missing values/(size of dataframe)
	No. of instances with missing values : Total no. of records with atleast one missing values
	Fraction of instances with missing values over all instances: No. of instances with missing values/(No. of instances)
4. Encoding the attributes
	-> Create a new column 'Missing' to keep track of instances with atleast one missing value
	-> Encode all the attributes(without the above column) into nominal variables using labelencoder.fit_transform(sklearn.preprocessing) that returns the encoded version of the data
	-> Print all the unique values of the dataset after encoding along with the encoded values
5. Decision Tree
	-> Read the target data for the decision tree
	-> Combine all the attributes to form a new dataframe
	-> Drop any missing values before building decision tree using the 'Missing column'
	-> Create training set and test set using train_test_split from sklearn.model_selection
	-> Create a decision tree for classifying an individual to one of the <= 50K and > 50K categories using tree.DecisionTreeClassifier from sklearn
	-> Fit the classifier using the training data that was generated by splitting the data
	-> Calculate the predicted Y value using the test data
	-> Calculate the Score and Error Rate of training as well as test data using inbuilt function of the classifier
	-> Calculate the accuracy score of the Decision Tree using sklearn.metrics
6. Decision Tree with Missing Data
	-> missing values = null 
	-> Construct a dataframe consisting atleast one missing value
	-> Construct a dataframe that does not contain any missing values
	-> Construct two datasets consisting all records with atleast one missing value and an equal no. of records with no missing value. 
		No. of missing records: 3620
		Length of each dataset: 7240
	-> Construct D1 by creating a new value “missing” for each attribute and using this value for every missing value in D	
	-> Construct D2 by using the most popular value for all missing values of each attribute
	-> Create test dataset with all the remaining data
	-> Encode the training and test datasets before creating decision trees
	-> Create and fit the Decision Tree for D1 and D2
	-> Calculate Error rate for D1 and D2 dataset using inbuilt function of the classifier(clf.score())
	-> Predict the value using the classifier
	-> Calculate the Accuracy Score of the Decision Tree using sklearn.metrics
	


**********************************************************************************************
****************************************CLUSTERING********************************************
**********************************************************************************************	

Files:
CW_Clustering.py
wholesale_customers.csv

Code Structure:

1. Read the below columns of the dataset wholesale_customers.csv into a dataframe
	-> ['Fresh','Milk','Grocery','Frozen','Detergents_Paper','Delicassen']
2. Create a list to print the mean and range of all the attributes
3. Loop through the columns/attributes of the dataframe,populate the above list and print the list of mean and range of each attributes
4. K Means Clustering:
	-> Construct the K-Means Clustering using sklearn.cluster.KMeans for k = 3
	-> Random state is taken as 0 so that the same set of result is obtained during each run
	-> The K-means clustering model is then fitted to the customer dataset comprising of six attributes
	-> As there are 6 attributes, due to the dimensionality issue, pairwise attributes are considered to graphically represent the fit of the model
	-> The graphical representation is done using 5*3 subplots to showcase the scatterplots for 15 pairs of attributes
5. K Means Clustering for different values of K:
	-> Create a list of values for K as below:
		K = [3,5,10]
	-> For each value of K, run the k-means and calculate the ratio of BC/WC	
	-> Calculation of BC : 
		-- The eucledian distance between the clusters is computed using 'euclidean_distances' built in function by passing the centers of the K Means Clusters
		-- The squared distance is added to compute the Between Cluster distance
	-> Calculation of WC :
		-- The within cluster distance is computed by using the built in component of K-Means model 'Inertia'
	-> The ratio of BC:WC is computed 	
	
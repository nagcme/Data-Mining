'''
References: https://scikit-learn.org/stable/modules/clustering.html#k-means
            https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.euclidean_distances.html
Assumptions:
            1. The input dataset 'adult.csv' is present in the same directory as the python file
'''

# Import statements
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
from sklearn import tree

# Part 1
# Classification
# Read adult.csv, only the required columns are imported in dataframe
adult = pd.read_csv('adult.csv',
                    usecols = ['age','workclass','education','education-num','marital-status','occupation',
                                'relationship','race','sex','capitalgain','capitalloss','hoursperweek',
                                'native-country'],
                    header=0)

# Question 1
print('***********************************************************')
print('Classification Answer 1:')
print('***********************************************************')
# number of instances
print('No. of instances: '+str(len(adult)))

# number of missing values
print('No. of missing values: '+str((adult.isnull().sum()).sum()))

# fraction of missing values over all attribute values
print('Fraction of missing values over all attributes: '+str(round(((adult.isnull().sum()).sum())/(adult.size),4)))

# number of instances with missing values
print('No. of instances with missing values: '+str(len((adult[adult.isnull().any(axis=1)]))))

# fraction of instances with missing values over all instances
print('Fraction of instances with missing values over all instances: '
      +str(round(len((adult[adult.isnull().any(axis=1)]))/len(adult),4)))

# Question 2
# Convert all 13 attributes into nominal using a Scikit-learn LabelEncoder.
# Then, print the set of all possible discrete values for each attribute.
print('***********************************************************')
print('Classification Answer 2:')
print('***********************************************************')
# Convert all 13 attributes into nominal
labelencoder = LabelEncoder()

# Add an attribute as missing value flag to keep track of the
# missing values after encoding
adult['Missing'] = adult.isnull().any(axis=1)

# Encode each attribute set and print the Unique classes(original values)
# and the encoded values of the dataset
for col in adult.columns:
    try:
        # Do not encode the 'Missing' column
        if col == 'Missing':
            break
        adult[col] = labelencoder.fit_transform(adult[col])
        print('Attribute: '+str(col))
        print('Unique Classes: ' + str(list(labelencoder.classes_)))
        print('Unique Values: ' + str(list(adult[col].unique())))
    except:
        # Handle if there are any errors while encoding
        print('Error in encoding the attribute: '+str(col))

# Question 3
# Ignore any instance with missing value(s) and use Scikit-learn to build a decision tree
# for classifying an individual to one of the <= 50K and > 50K categories. Compute the error
# rate of the resulting tree
print('***********************************************************')
print('Classification Answer 3:')
print('***********************************************************')

# Read the target data for the decision tree
adult_target = pd.read_csv('adult.csv',
                    usecols = ['class'],
                    header=0)

# combine all the attributes to form a new dataframe
adult_df_l = [adult,adult_target]
adult_df = pd.concat(adult_df_l,axis=1)

# Drop any missing values before building decision tree
adult_no_na = adult_df[adult_df['Missing'] == False]

# Create training set and test set
X_train,X_test,y_train,y_test=train_test_split(adult_no_na.iloc[:,0:13],
                                               adult_no_na.iloc[:,14],
                                               random_state=0)

# Create a decision tree for classifying an individual
# to one of the <= 50K and > 50K categories
clf = tree.DecisionTreeClassifier(random_state=0)

# Fit the classifier using the training data
clf.fit(X_train,y_train)

# Calculate the predicted Y value
y_hat = clf.predict(X_test)

# Calculate Error rate
print('Score (Test Data) :'+str(round((clf.score(X_test,y_test)),3)))
print('Error Rate (Test Data) :'+str(round((1-clf.score(X_test,y_test)),3)))
print('Score (Training Data) :'+str(round((clf.score(X_train,y_train)),3)))
print('Error Rate (Training Data) :'+str(round((1-clf.score(X_train,y_train)),3)))

# Calculate the accuracy score of the Decision Tree
acc_scr = metrics.accuracy_score(y_test, y_hat)
print('Accuracy score:', str(round(acc_scr,2)))
print('Confusion matrix:')
print(metrics.confusion_matrix(y_test, y_hat))
print("Classification Report")
print(metrics.classification_report(y_test, y_hat))

# Question 4
# all instances with at least one missing value
# Assumption: missing values = null hence checking with isnull()
print('***********************************************************')
print('Classification Answer 4:')
print('***********************************************************')

# Read adult.csv, only the required columns are imported in dataframe
adult = pd.read_csv('adult.csv',
                    usecols = ['age','workclass','education','education-num','marital-status','occupation',
                                'relationship','race','sex','capitalgain','capitalloss','hoursperweek',
                                'native-country','class'],
                    header=0)

# construct a dataframe consisting atleast one missing value
df_miss = adult[adult.isnull().any(axis=1) == True]
# construct a dataframe that does not contain any missing values
df_no_miss = adult[adult.isnull().any(axis=1) == False]

# Construct two datasets consisting all records with atleast one missing value
# and an equal no. of records with no missing value
df_no_miss_eq = df_no_miss.head(len(df_miss))
df_l = [df_miss,df_no_miss_eq]
D1 = pd.concat(df_l)
D2 = pd.concat(df_l)

# construct D1 by creating a new value “missing” for each attribute and using this value for
# every missing value in D
D1.fillna('Missing',inplace=True)

# construct D2 by using the most popular value for all missing values of each attribute
D2 = D2.fillna(adult.mode().iloc[0])

# Create test dataset with all the remaining data
D_test = df_no_miss.tail(len(df_no_miss)-len(df_miss)).copy()

# Encode the training and test datasets before creating decision trees
for col in D1.columns:
    try:
        D1[col] = labelencoder.fit_transform(D1[col])
        D2[col] = labelencoder.fit_transform(D2[col])
        D_test[col] = labelencoder.fit_transform(D_test[col])
    except:
        print('Error in encoding the attribute: '+str(col))

# Create and fit the Decision Tree for D1 and D2
clf1 = tree.DecisionTreeClassifier(random_state=0)
clf1.fit(D1.iloc[:,0:13],D1.iloc[:,13])

clf2 = tree.DecisionTreeClassifier(random_state=0)
clf2.fit(D2.iloc[:,0:13],D2.iloc[:,13])

# Calculate Error rate for D1 dataset
print('****************************')
print('*********D1 Dataset*********')
print('****************************')
print('Score (Main Dataset) : '+str(round((clf1.score(D1.iloc[:,0:13],D1.iloc[:,13])),3)))
print('Error Rate (Main Dataset) : '+str(round((1-clf1.score(D1.iloc[:,0:13],D1.iloc[:,13])),3)))
print('Score (Test Dataset) : '+str(round((clf1.score(D_test.iloc[:,0:13],D_test.iloc[:,13])),3)))
print('Error Rate (Test Dataset) : '+str(round((1-clf1.score(D_test.iloc[:,0:13],D_test.iloc[:,13])),3)))

# Predict the value using the classifier
y1_hat = clf1.predict(D_test.iloc[:,0:13])

# Accuracy Score
acc_scr = metrics.accuracy_score(D_test.iloc[:,13], y1_hat)
print('Accuracy score: '+str(round(acc_scr,2)))
print('Confusion matrix:')
print(metrics.confusion_matrix(D_test.iloc[:,13], y1_hat))

# Calculate Error rate for D2 dataset
print('****************************')
print('*********D2 Dataset*********')
print('****************************')
print('Score (Main Dataset) :'+str(round((clf2.score(D2.iloc[:,0:13],D2.iloc[:,13])),3)))
print('Error Rate (Main Dataset) :'+str(round((1-clf2.score(D2.iloc[:,0:13],D2.iloc[:,13])),3)))
print('Score (Test Dataset) :'+str(round((clf2.score(D_test.iloc[:,0:13],D_test.iloc[:,13])),3)))
print('Error Rate (Test Dataset) :'+str(round((1-clf2.score(D_test.iloc[:,0:13],D_test.iloc[:,13])),3)))
y2_hat = clf2.predict(D_test.iloc[:,0:13])

# Accuracy Score
acc_scr = metrics.accuracy_score(D_test.iloc[:,13], y2_hat)
print('Accuracy score: '+str(round(acc_scr,2)))
print('Confusion matrix:')
print(metrics.confusion_matrix(D_test.iloc[:,13], y2_hat))
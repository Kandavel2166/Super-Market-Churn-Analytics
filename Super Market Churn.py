#!/usr/bin/env python
# coding: utf-8

# In[45]:


# Import necessary packages to read, process and visualize data
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt     # Generate plots
import seaborn as sns               # Visualization
get_ipython().run_line_magic('matplotlib', 'inline')

# Read the data
filename = "D:/churn/Super Market Churn/Super-Market-Data.csv"
data = pd.read_csv(filename) 

# Let us see the shape of data
print(f"(Rows , Coloms):{data.shape}")   
# Following output shows there are 7043 rows and 21 columns in our data-9//


# In[ ]:





# In[ ]:





# In[46]:


# Overview and statistical details of the data..
# Let us see first five rows to understand what type of values exist for each columns
data.head()  #will display first 5 rows 
# data.head(10)


# In[ ]:





# In[ ]:





# In[47]:


# To view all column names and their respective data types
data.columns
print(" ")
data.info()    #To view all column names and their respective data types
print(" ")
data.describe() # Shows statistical summaries for all numeric columns


# From above output we can observe :
# *  Mean Monthly charges is about 64.76 units and 75% of observations are monthly charged around 89.85
# *  The maximum tenure is 72 months with mean being about 32 months.
# *  About 50% of customers stayed for 29 months tenure and were charged 70.3 per month  
# To get more relevant information, we will visualize attributes of the data and distribution of target variable(Churn)

# In[ ]:





# In[ ]:





# In[48]:


# Plot distribution of dependent/target variable - Churn column
data['Churn'].value_counts().head().plot.bar()     # To generate a bar plot

# To generate a pie chart. Since there are only two classes, a pie chart may look more appealing
sizes = data['Churn'].value_counts(sort = True)
labels = np.unique(data.Churn)

# Visualize the data
plt.figure(figsize = (6,6))   #size of pie figure
plt.subplot(121)
plt.title("\n\n\n  Customer churn rate:")
plt.pie(sizes, labels = labels, autopct='%1.1f%%')



# Bar & pie plots below show that number of customers churned is less than half of not churned. 


# In[ ]:





# In[ ]:





# ValueError Traceback (most recent call last) in 1 # Convert following object type columns to numeric ----> 2 data.TotalCharges = pd.to_numeric(data.TotalCharges, errors = '')
# 
# D:\Jupiter Notebooks\lib\site-packages\pandas\core\tools\numeric.py in to_numeric(arg, errors, downcast) 113 114 if errors not in ("ignore", "raise", "coerce"): --> 115 raise ValueError("invalid error value specified") 116 117 is_series = False
# 
# ValueError: invalid error value specified

# In[49]:


# Convert following object type columns to numeric        
data.TotalCharges = pd.to_numeric(data.TotalCharges, errors = 'coerce')


# In[50]:


# Let us find if there are any missing values in our data.
print("No.of missing values: \n",data.isnull().sum())


# Output shows that there are 11 total missing values in TotalCharges column.

# In[ ]:





# In[ ]:





# In[51]:


# Drop CustomerId column as it is not required
data.drop(['customerID'], axis = 1, inplace = True)

# Fill the missing values with 0
data['TotalCharges'] = data['TotalCharges'].fillna(0.0)

# Check for any existing missing values
print("Missing values now: \n", data.isnull().sum())


# Missing values for all columns are now 0. So, no more missing data.

# In[ ]:





# In[52]:


data.head(50)


# In[53]:


# Now let us work on categorical features.                  
data.gender = [1 if x == "Male" else 0 for x in data.gender]
for col in ('Membership', 'Dependents', 'Walk in Store' , 'Vegetables',
        'Dairy Products','Organic Oils', 'Freezer Essentials','Cereals',
        'Fruits','PaperlessBilling','Online order','Churn'):
    data[col] = [1 if x == "Yes" else 0 for x in data[col]]        
data.head(10)   # See how data looks like now


# In[ ]:





# In[ ]:





# # Correlation:

# Now, let us see which features are most effective in causing customer churn.
# **Correlation -**
# Correlation between variables shows how dependent variable changes due to an independent variable under consideration. 
# A value close to +1 signifies strong positive correlation, while close to -1 shows strong negative effect. Correlation coeff. close to zero signifies weak relation between features.

# Now, let us see which features are most effective in causing customer churn.

# **Correlation -**    Correlation between variables shows how dependent variable changes due to an independent variable under consideration. 
# A value close to +1 signifies strong positive correlation, while close to -1 shows strong negative effect. Correlation coeff. close to zero signifies weak relation between features. 

# _______________________________________________________________________________________________________________________________

# **Pearson Correlation Coefficient :**

# +1 = Positive Correlation
# 

# 0 = No Correlation

# -1 = Negative Correlation

# ______________________________________________________________________________________________________________________________

# Default method in correlation is known as **Pearson** 

# **Pearson**  is standard correlation coefficient

# Pearson's correlation coefficient is the test statistics that measures the statistical relationship, or association, between two continuous variables

# In[60]:


# Print correlation between all features and target variable  
data.corr()['Churn'].sort_values()


# In[ ]:





# # Seaborn Correlation Heatmap :
# 

# In[61]:


# Plot heatmap using Seaborn to visualize correlation amongst ftrs.
plt.figure(figsize = (16,16))
sns.heatmap(data.corr(), annot = True)


# In[66]:


# For following features, let us generate bar plots w.r.t. target variable
for col in ('Membership', 'Dependents', 'Walk in Store' , 'Vegetables',
        'Dairy Products','Organic Oils', 'Freezer Essentials','Cereals',
        'Fruits','PaperlessBilling','Online order'):

    sns.barplot(x = col, y = 'Churn' , data = data )
    plt.show()
# Following plots show Churn rate for each category of these categorical features.


# In[ ]:





# In[67]:


# Generate pairplots for all features.
highCorrCols = ['MonthlyCharges','TotalCharges','tenure','Churn']
sns.pairplot(data[highCorrCols], hue = 'Churn')


# In[ ]:





# In[68]:


# Prepare data for model training and testing input.
y = data.Churn.values     # Target feature

# All features except class (target)
data = pd.get_dummies(data)
X = data.drop(["Churn","gender"],axis=1)

from sklearn.metrics import accuracy_score, mean_squared_error as mse
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.linear_model import LogisticRegression as LR
from sklearn.svm import SVC 
from sklearn.neural_network import MLPClassifier

# Split the data into training and testing data
X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size = 0.2, random_state=1)

# Classification using RBF SVM  
svc_rbf = SVC(kernel = "rbf")
svc_rbf = svc_rbf.fit(X_train,y_train)
prediction = svc_rbf.predict(X_test)
print("Mean-squared error using SVM RBF:", mse(y_test, prediction))
print("Accuracy with SVM RBF:",accuracy_score(y_test, prediction))

# Classification using Random Forest Classifier
rfc = RF(max_depth= 5, n_estimators= 10, max_features= 'auto')
rfc = rfc.fit(X_train,y_train)
prediction = rfc.predict(X_test)
print("Mean-squared error using Random Forest Classifier:", mse(y_test, prediction))
print("Accuracy with Random Forest Classifier:",accuracy_score(y_test, prediction))

# Classification using Logistic Regression
logreg = LR(C = 1)
logreg = logreg.fit(X_train,y_train)
prediction = logreg.predict(X_test)
print("Mean-squared error using Logistic Regression:", mse(y_test, prediction))
print("Accuracy with Logistic Regression:",accuracy_score(y_test, prediction))

# Classification using Multi-layer perceptron 
ann = MLPClassifier(solver='lbfgs', alpha = 1e-5,
                    hidden_layer_sizes = (5, 2), random_state = 1)
ann = ann.fit(X_train, y_train)
prediction = ann.predict(X_test)
print("Mean-squared error using Neural networks MLP:", mse(y_test, prediction))
print("Accuracy with Neural networks MLP:",accuracy_score(y_test, prediction))


# In[ ]:





# In[69]:


(X_train)


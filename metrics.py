# https://www.datacamp.com/community/tutorials/decision-tree-classification-python
# https://app.datacamp.com/workspace/w/01832fe5-bf76-4856-bea4-dd8bd8462d8e/edit

# Load libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

# col_names = ['price_1', 'price_2', 'order_large', 'order_test', 'additional_service', 'order_confirm', 'operator_request', 'delivery_address', 'One_click_order']
# # load dataset
# pima = pd.read_csv("dataset_new.csv", header=None, names=col_names)
# # pima-indians-diabetes


col_names = ['price_1', 'price_2', 'order_large', 'order_test', 'additional_service', 'order_confirm', 'operator_request', 'delivery_address', 'One_click_order']
# load dataset
pima = pd.read_csv("dataset.csv", header=None, names=col_names)

#split dataset in features and target variable
feature_cols = ['price_1', 'price_2', 'order_large', 'order_test','additional_service','order_confirm','operator_request']
X = pima[feature_cols] # Features
y = pima.One_click_order # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test|

# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

from sklearn import tree
import matplotlib.pyplot as plt

plt.figure(figsize=(290, 150), dpi=70)
tree.plot_tree(clf,
               feature_names=feature_cols,
               class_names=["0", "1"],
               filled=True,
               rounded=True);
plt.savefig('dtree.png')


# Create Decision Tree classifer object
clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

plt.figure(figsize=(80, 30), dpi=80)
tree.plot_tree(clf,
               feature_names=feature_cols,
               class_names=["0", "1"],
               filled=True,
               rounded=True);
plt.savefig('orders-tree.png')
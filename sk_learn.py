import time
from sklearn import metrics
import numpy as np
import pandas as pd

csv = []
f = pd.read_pickle("/Users/neo/Downloads/mnist.pkl")
train,val,test = f
train_x = train[0]  
train_y = train[1]  
test_x = test[0]  
test_y = test[1]
thresh = 0.5  
model_save_file = None  
model_save = {}
# Multinomial Naive Bayes Classifier
def naive_bayes_classifier(train_x, train_y):
    from sklearn.naive_bayes import MultinomialNB
    model = MultinomialNB(alpha=0.01)
    model.fit(train_x, train_y)
    return model


# KNN Classifier
def knn_classifier(train_x, train_y):
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier()
    model.fit(train_x, train_y)
    return model


# Logistic Regression Classifier
def logistic_regression_classifier(train_x, train_y):
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(penalty='l2')
    model.fit(train_x, train_y)
    return model


# Random Forest Classifier
def random_forest_classifier(train_x, train_y):
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=8)
    model.fit(train_x, train_y)
    return model


# Decision Tree Classifier
def decision_tree_classifier(train_x, train_y):
    from sklearn import tree
    model = tree.DecisionTreeClassifier()
    model.fit(train_x, train_y)
    return model


# GBDT(Gradient Boosting Decision Tree) Classifier
def gradient_boosting_classifier(train_x, train_y):
    from sklearn.ensemble import GradientBoostingClassifier
    model = GradientBoostingClassifier(n_estimators=200)
    model.fit(train_x, train_y)
    return model


# SVM Classifier
def svm_classifier(train_x, train_y):
    from sklearn.svm import SVC
    model = SVC(kernel='rbf', probability=True)
    model.fit(train_x, train_y)
    return model

test_classifiers = ['NB','DT']
classifiers = {'NB':naive_bayes_classifier, 
                  'KNN':knn_classifier,
                   'LR':logistic_regression_classifier,
                   'RF':random_forest_classifier,
                   'DT':decision_tree_classifier,
                  'SVM':svm_classifier,
                #'SVMCV':svm_cross_validation,
                 'GBDT':gradient_boosting_classifier
    }
print ('reading training and testing data...')  
num_train, num_feat = train_x.shape  
num_test, num_feat = test_x.shape  
is_binary_class = (len(np.unique(train_y)) == 2)  
print ('******************** Data Info *********************')
print ('#training data: %d, #testing_data: %d, dimension: %d' % (num_train, num_test, num_feat))
for classifier in test_classifiers:  
    print ('******************* %s ********************' % classifier)
    csv.append('******************* %s ********************' % classifier)
    start_time = time.time()  
    model = classifiers[classifier](train_x, train_y)
    print ('training took %fs!' % (time.time() - start_time))
    csv.append('training took %fs!' % (time.time() - start_time))
    predict = model.predict(test_x)
    if model_save_file != None:  
        model_save[classifier] = model  
    if is_binary_class:  
        precision = metrics.precision_score(test_y, predict)  
        recall = metrics.recall_score(test_y, predict)
        print ('precision: %.2f%%, recall: %.2f%%' % (100 * precision, 100 * recall))
        csv.append('precision: %.2f%%, recall: %.2f%%' % (100 * precision, 100 * recall))
    accuracy = metrics.accuracy_score(test_y, predict)
    print ('accuracy: %.2f%%' % (100 * accuracy))
    csv.append('accuracy: %.2f%%' % (100 * accuracy))
    dcsv = pd.DataFrame(csv)
    dcsv.to_csv("/Users/neo/java/aa.csv",index = False,encoding = 'utf-8')
#by neo

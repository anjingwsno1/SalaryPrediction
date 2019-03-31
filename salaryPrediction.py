import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import *
from sklearn.metrics import *
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import svm
from sklearn import tree
from sklearn.preprocessing import *
from sklearn.model_selection import validation_curve
from sklearn.decomposition import PCA 
import time

class DATA:
    def __init__(self):
        self.train_data = []
        self.train_label = []
        self.fp_rate = []
        self.tp_rate = []

def init(DATA):
    train_file = pd.read_csv('adult/adult.data.csv',header=None,sep=r'\s*,\s*',engine='python',na_values='?')
    # train_file = pd.read_csv('adult/adult.data.csv',header=None)
    train_file.columns = ['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','salary']
##    delete unused colums
    #del train_file['workclass']
    del train_file['education']
    del train_file['native-country']
    del train_file['capital-gain']
    del train_file['capital-loss']
# delete unknow rows
    #train_file = train_file.dropna()
#    feature extraction
    encodedData = {}
    for eachColumn in train_file.columns:
        if train_file.dtypes[eachColumn] == np.object:
            encodedData[eachColumn] = LabelEncoder()
            train_file[eachColumn] = encodedData[eachColumn].fit_transform(train_file[eachColumn])

    DATA.train_data = scale(train_file.iloc[:,:-1])
    DATA.train_label = train_file['salary'].values
# replace the missing value with mean/most_frequent values
    '''
    imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
    #imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    imp.fit(DATA.train_data)
    DATA.train_data = imp.transform(DATA.train_data)
    '''
# using PCA to decompose
'''
    pca = PCA(n_components=7)
    DATA.train_data = pca.fit_transform(DATA.train_data)
'''
def NB(DATA):
    nb = GaussianNB()
    start = time.time()
    scores = cross_val_score(nb,DATA.train_data,DATA.train_label,cv=10,scoring = 'accuracy')
    end = time.time()
    x_train,x_test,y_train,y_test = train_test_split(DATA.train_data,DATA.train_label,test_size = 0.1)
    nb.fit(x_train,y_train)
    prediction = nb.predict(x_test)
    proba = nb.predict_proba(x_test)[:,1]
    fp_rate,tp_rate,thresold = roc_curve(y_test,proba)
    DATA.fp_rate.append(fp_rate)
    DATA.tp_rate.append(tp_rate)
    print('Naive Bayers:')
    print('accuracy =',scores.mean())
    print('precision score =',precision_score(y_test,prediction,average='macro'))
    print('recall score =',recall_score(y_test,prediction,average='macro'))
    print('f1 score =',f1_score(y_test,prediction,average='macro'))
    print('runing time =',end-start)
    
def LR(DATA):
    lr = LogisticRegression()
    start = time.time()
    scores = cross_val_score(lr,DATA.train_data,DATA.train_label,cv=10,scoring = 'accuracy')
    end = time.time()
    x_train,x_test,y_train,y_test = train_test_split(DATA.train_data,DATA.train_label,test_size = 0.1)
    lr.fit(x_train,y_train)
    prediction = lr.predict(x_test)
    proba = lr.predict_proba(x_test)[:,1]
    fp_rate,tp_rate,thresold = roc_curve(y_test,proba)
    DATA.fp_rate.append(fp_rate)
    DATA.tp_rate.append(tp_rate)
    print('Logistic Regression:')
    print('accuracy =',scores.mean())
    print('precision score =',precision_score(y_test,prediction,average='macro'))
    print('recall score =',recall_score(y_test,prediction,average='macro'))
    print('f1 score =',f1_score(y_test,prediction,average='macro'))
    print('runing time =',end-start)

def KNN(DATA):
    '''
    y = []
    for i in range(1,51):
        knn = KNeighborsClassifier(n_neighbors=i)
        scores = cross_val_score(knn,DATA.train_data,DATA.train_label,cv=10,scoring = 'accuracy')
        y.append(scores.mean())
    #plt.style.use('ggplot')
    plt.plot(range(1,51),y)
    plt.xlabel('Values of K for KNN')
    plt.ylabel('Cross-Validated Accuracy')
    plt.show()
    '''
    
    knn = KNeighborsClassifier(n_neighbors=25)
    start = time.time()
    scores = cross_val_score(knn,DATA.train_data,DATA.train_label,cv=10,scoring = 'accuracy')
    end = time.time()
    x_train,x_test,y_train,y_test = train_test_split(DATA.train_data,DATA.train_label,test_size = 0.1)
    knn.fit(x_train,y_train)
    prediction = knn.predict(x_test)
    proba = knn.predict_proba(x_test)[:,1]
    fp_rate,tp_rate,thresold = roc_curve(y_test,proba)
    DATA.fp_rate.append(fp_rate)
    DATA.tp_rate.append(tp_rate)
    print('K-NN:')
    print('accuracy =',scores.mean())
    print('precision score =',precision_score(y_test,prediction,average='macro'))
    print('recall score =',recall_score(y_test,prediction,average='macro'))
    print('f1 score =',f1_score(y_test,prediction,average='macro'))
    print('runing time =',end-start)

def RF(DATA):
    rf = RandomForestClassifier(n_estimators=100)
    start = time.time()
    scores = cross_val_score(rf,DATA.train_data,DATA.train_label,cv=10,scoring = 'accuracy')
    end = time.time()
    x_train,x_test,y_train,y_test = train_test_split(DATA.train_data,DATA.train_label,test_size = 0.1)
    rf.fit(x_train,y_train)
    prediction = rf.predict(x_test)
    proba = rf.predict_proba(x_test)[:,1]
    fp_rate,tp_rate,thresold = roc_curve(y_test,proba)
    DATA.fp_rate.append(fp_rate)
    DATA.tp_rate.append(tp_rate)
    print('Random Forest:')
    print('accuracy =',scores.mean())
    print('precision score =',precision_score(y_test,prediction,average='macro'))
    print('recall score =',recall_score(y_test,prediction,average='macro'))
    print('f1 score =',f1_score(y_test,prediction,average='macro'))
    print('runing time =',end-start)

def SVC(DATA):
    '''
    param_range = np.linspace(0.001,0.3,15)
    train_loss,test_loss = validation_curve(svm.SVC(),DATA.train_data,DATA.train_label,'gamma',param_range,scoring='neg_mean_squared_error')
    train_loss = -np.mean(train_loss,axis=1)
    test_loss = -np.mean(test_loss,axis=1)
    plt.plot(param_range,train_loss,label='Training')
    plt.plot(param_range,test_loss,label='cross_validation')
    plt.xlabel('gamma')
    plt.ylabel('loss')
    plt.legend(loc='best')
    plt.show()
    '''
    svc = svm.SVC(gamma=0.01)
    start = time.time()
    scores = cross_val_score(svc,DATA.train_data,DATA.train_label,cv=10,scoring = 'accuracy')
    end = time.time()
    x_train,x_test,y_train,y_test = train_test_split(DATA.train_data,DATA.train_label,test_size = 0.1)
    svc.fit(x_train,y_train)
    prediction = svc.predict(x_test)
    proba = svc.decision_function(x_test)
    fp_rate,tp_rate,thresold = roc_curve(y_test,proba)
    DATA.fp_rate.append(fp_rate)
    DATA.tp_rate.append(tp_rate)
    print('SVC:')
    print('accuracy =',scores.mean())
    print('precision score =',precision_score(y_test,prediction,average='macro'))
    print('recall score =',recall_score(y_test,prediction,average='macro'))
    print('f1 score =',f1_score(y_test,prediction,average='macro'))
    print('runing time =',end-start)

def AB(DATA):
    ab = AdaBoostClassifier()
    start = time.time()
    scores = cross_val_score(ab,DATA.train_data,DATA.train_label,cv=10,scoring = 'accuracy')
    end = time.time()
    x_train,x_test,y_train,y_test = train_test_split(DATA.train_data,DATA.train_label,test_size = 0.1)
    ab.fit(x_train,y_train)
    prediction = ab.predict(x_test)
    proba = ab.predict_proba(x_test)[:,1]
    fp_rate,tp_rate,thresold = roc_curve(y_test,proba)
    DATA.fp_rate.append(fp_rate)
    DATA.tp_rate.append(tp_rate)
    print('AdaBoost:')
    print('accuracy =',scores.mean())
    print('precision score =',precision_score(y_test,prediction,average='macro'))
    print('recall score =',recall_score(y_test,prediction,average='macro'))
    print('f1 score =',f1_score(y_test,prediction,average='macro'))
    print('runing time =',end-start)
    
if __name__ == '__main__':
    data = DATA()
    init(data)
    NB(data)
    KNN(data)
    LR(data)
    RF(data)
    SVC(data)
    AB(data)

# '''
    plt.figure()
    # plt.style.use('ggplot')
    plt.plot(data.fp_rate[0],data.tp_rate[0],label='NB')
    plt.plot(data.fp_rate[1],data.tp_rate[1],label='KNN')
    plt.plot(data.fp_rate[2],data.tp_rate[2],label='LR')
    plt.plot(data.fp_rate[3],data.tp_rate[3],label='RF')
    plt.plot(data.fp_rate[4],data.tp_rate[4],label='SVC')
    plt.plot(data.fp_rate[5],data.tp_rate[5],label='AB')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.grid(True)
    plt.legend()
    plt.show()
# ''' 

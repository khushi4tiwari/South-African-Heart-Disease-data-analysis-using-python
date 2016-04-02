import numpy as np
import xlrd
import sklearn.linear_model as lm
from pylab import *
from methods import *
from scipy.stats import zscore
import scipy

class ResultHolder:
    resultNo = 0
    size = 1
    results = np.zeros(0)
    
    def __init__(self, iterations):
        self.results = np.zeros(iterations)
        self.size = iterations
        self.resultNo = 0
        
    def addResult(self, res):
        self.results[self.resultNo] = res
        self.resultNo += 1
        
    def getMeanResult(self):
        sum = 0.0
        for i in range(0,self.size):
            sum += self.results[i]
        return double(sum) / double(self.size)
        
    def getResults(self):
        res = np.zeros(self.size+1)
        for i in range(0,self.size):
            res[i] = self.results[i]
        res[self.size] = self.getMeanResult()
        return res

#Converts Present and Absent into numbers.
def convert(s):
    if s == "Present":
        return 1
    else:
        return 0

#Load dataset
doc = xlrd.open_workbook('../../dataset_sorted.xls').sheet_by_index(0)

size = 463
noAttributes = 9

#Get attributes and classnames
attributeNames = doc.row_values(0,1,noAttributes+1)
attributeNamesCHD = doc.row_values(0,1,noAttributes+1+1)

classLabels = doc.col_values(noAttributes+1,1,size)
classNames = sorted(set(classLabels))
classDict = dict(zip(classNames,range(2)))


y = np.mat([classDict[value] for value in classLabels]).T

X = np.mat(np.empty((size-1,noAttributes)))
XCHD =np.mat(np.empty((size-1,noAttributes+1)))

for i, col_id in enumerate(range(1,noAttributes+1+1)):
    if(i < len(attributeNames) and attributeNames[i] == "famhist"):
        temp12 = [convert(i2) for i2 in doc.col_values(col_id,1,size)]
        if i < noAttributes:
            X[:,i] = np.mat(temp12).T
        XCHD[:,i] = np.mat(temp12).T
    else:
        if i < noAttributes:
            X[:,i] = np.mat(doc.col_values(col_id,1,size)).T
        XCHD[:,i] = np.mat(doc.col_values(col_id,1,size)).T

M = len(attributeNames) 
N = len(y)
C = len(classNames)


XStandardized = zscore(X, ddof=1)

XPC = getPrincipalComponents(XStandardized)

(X_train,y_train),(X_test,y_test) = getTestAndTrainingSet(X,y)
(X_train_PC,y_train_PC),(X_test_PC,y_test_PC) = getTestAndTrainingSet(XStandardized,y)

forwardSelection(X,y,N,5,attributeNames,classNames)

artificialNeuralNetwork(X,y,N,noAttributes)

artificialNeuralNetworkByPC(XStandardized,y,N)

Xad = np.copy(X)

#Xad = scipy.delete(Xad,8,1) # Age
#Xad = scipy.delete(Xad,7,1) # Alcohol
Xad = scipy.delete(Xad,6,1) # Obesity
#Xad = scipy.delete(Xad,5,1) # TypeA
#Xad = scipy.delete(Xad,4,1) # Famhist
Xad = scipy.delete(Xad,3,1) # Adiposity
#Xad = scipy.delete(Xad,2,1) # LDL
Xad = scipy.delete(Xad,1,1) # Tobacco
Xad = scipy.delete(Xad,0,1) # SBP

X2PC = np.copy(XPC)
X2PC = X2PC[:,0:2]

predictLinearRegression(Xad,y)

(_,_,attributeNamesXad) = removeAttribute(X,y,6,attributeNames)
(_,_,attributeNamesXad) = removeAttribute(X,y,3,attributeNamesXad)
(_,_,attributeNamesXad) = removeAttribute(X,y,1,attributeNamesXad)
(_,_,attributeNamesXad) = removeAttribute(X,y,0,attributeNamesXad)

PCNames = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9']

(_,_,attributeNamesX2PC) = removeAttribute(X,y,8,PCNames)
(_,_,attributeNamesX2PC) = removeAttribute(X,y,7,attributeNamesX2PC)
(_,_,attributeNamesX2PC) = removeAttribute(X,y,6,attributeNamesX2PC)
(_,_,attributeNamesX2PC) = removeAttribute(X,y,5,attributeNamesX2PC)
(_,_,attributeNamesX2PC) = removeAttribute(X,y,4,attributeNamesX2PC)
(_,_,attributeNamesX2PC) = removeAttribute(X,y,3,attributeNamesX2PC)
(_,_,attributeNamesX2PC) = removeAttribute(X,y,2,attributeNamesX2PC)


artificialNeuralNetwork(Xad, y, N, noAttributes-4)


# Classification
s1 = "X not modified."
s2 = "Attributes of X selected according to result of forward selection."
s3 = "X represented by principal components"
s4 = "X represented by two most important principal components."


logisticRegression(X,y, s=s1)
logisticRegression(Xad,y, s=s2)
logisticRegression(XPC,y, s=s3)
logisticRegression(X2PC,y, s=s4)
    
decisionTree(X,y,attributeNames,classNames,"Decision_Tree_X.gvz",s=s1)
decisionTree(Xad,y,attributeNamesXad,classNames,"Decision_Tree_Xad.gvz",s=s2)
decisionTree(XPC,y,PCNames,classNames,"Decision_Tree_XPC.gvz",s=s3)
decisionTree(X2PC,y,attributeNamesX2PC,classNames,"Decision_Tree_X2PC.gvz",s=s4)
   

(XK,e1) = kNearestNeighbours(X,y,C,s=s1)
(XKad,e2) = kNearestNeighbours(Xad,y,C,s=s2)
(XKPC,e3) = kNearestNeighbours(XPC,y,C,s=s3)
(XK2PC,e4) = kNearestNeighbours(X2PC,y,C,s=s4)

#iterations = 5
#
#lrH = ResultHolder(iterations)
#lrHad = ResultHolder(iterations)
#lrHPC = ResultHolder(iterations)
#lrH2PC = ResultHolder(iterations)
#
#dcH = ResultHolder(iterations)
#dcHad = ResultHolder(iterations)
#dcHPC = ResultHolder(iterations)
#dcH2PC = ResultHolder(iterations)
#
#knH = ResultHolder(iterations)
#knHad = ResultHolder(iterations)
#knHPC = ResultHolder(iterations)
#knH2PC = ResultHolder(iterations)
#
#for i in range(0,iterations):
#    (X_train,y_train),(X_test,y_test) = getTestAndTrainingSet(X,y)
#    (X_train_PC,y_train_PC),(X_test_PC,y_test_PC) = getTestAndTrainingSet(XStandardized,y)
#
#    (X_train_ad,y_train_ad),(X_test_ad,y_test_ad) = getTestAndTrainingSet(Xad,y)
#    (X_train_2PC,y_train_2PC),(X_test_2PC,y_test_2PC) = getTestAndTrainingSet(X2PC,y)
#
#    lrH.addResult(logisticRegression(X,y,X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test, s=s1))
#    lrHad.addResult(logisticRegression(Xad,y,X_train = X_train_ad, y_train = y_train_ad, X_test = X_test_ad, y_test = y_test_ad, s=s2))
#    lrHPC.addResult(logisticRegression(XPC,y,X_train = X_train_PC, y_train = y_train_PC, X_test = X_test_PC, y_test = y_test_PC, s=s3))
#    lrH2PC.addResult(logisticRegression(X2PC,y,X_train = X_train_2PC, y_train = y_train_2PC, X_test = X_test_2PC, y_test = y_test_2PC, s=s4))
#    
#    dcH.addResult(decisionTree(X,y,attributeNames,classNames,"Decision_Tree_X.gvz",s=s1, X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test))
#    dcHad.addResult(decisionTree(Xad,y,attributeNamesXad,classNames,"Decision_Tree_Xad.gvz",s=s1,X_train = X_train_ad,y_train = y_train_ad,X_test = X_test_ad,y_test = y_test_ad))
#    dcHPC.addResult(decisionTree(XPC,y,PCNames,classNames,"Decision_Tree_XPC.gvz",s=s1,X_train = X_train_PC,y_train = y_train_PC,X_test = X_test_PC,y_test = y_test_PC))
#    dcH2PC.addResult(decisionTree(X2PC,y,attributeNamesX2PC,classNames,"Decision_Tree_X2PC.gvz",s=s1,X_train = X_train_2PC,y_train = y_train_2PC,X_test = X_test_2PC, y_test = y_test_2PC))
#    
#    knH.addResult(plotKNearestNeighbours(classNames, X, y, C, s=s1, K=15,X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test))
#    knHad.addResult(plotKNearestNeighbours(classNames, Xad, y, C, s=s2, K=14,X_train = X_train_ad, y_train = y_train_ad, X_test = X_test_ad, y_test = y_test_ad))
#    knHPC.addResult(plotKNearestNeighbours(classNames, XPC, y, C, s=s3, K=24,X_train = X_train_PC, y_train = y_train_PC, X_test = X_test_PC, y_test = y_test_PC))
#    knH2PC.addResult(plotKNearestNeighbours(classNames, X2PC, y, C, DoPrincipalComponentAnalysis = True,s=s4,K=30,X_train = X_train_2PC, y_train = y_train_2PC, X_test = X_test_2PC, y_test = y_test_2PC))
#
#print lrH.getResults()
#print lrHad.getResults()
#print lrHPC.getResults()
#print lrH2PC.getResults()
#
#print dcH.getResults()
#print dcHad.getResults()
#print dcHPC.getResults()
#print dcH2PC.getResults()
#
#print knH.getResults()
#print knHad.getResults()
#print knHPC.getResults()
#print knH2PC.getResults()

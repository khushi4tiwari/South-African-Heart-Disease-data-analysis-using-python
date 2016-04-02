from pylab import *
from scipy.stats import zscore
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import xlrd

#Converts Present and Absent into numbers.
def convert(s):
    if s == "Present":
        return 1
    else:
        return 0

#Converting for legend
def convertToWord(n):
   if (n < 0.5):
        return "Negative CHD"
   else:
        return "Positive CHD"
        
        
#Calculate mean and variance and range
def calculateStatistics(X,noAttributes):
    #res = np.matrix
    for i in range(noAttributes):
        temp = []
        x = X[:,i]
        #print(x)
        temp.append(x.mean())
        temp.append(x.var(ddof=1))
        #temp.append(np.median(x))
        temp.append(x.max())
        temp.append(x.min())
        print(temp)
    #return res

#Create boxplots of the nine attributes 
def boxPlot(X,attributeNames):
    fig = figure()
    fig.subplots_adjust(hspace=.3)
    for i in range(0,9):
        subplot(3,3,i+1)
        boxplot(X[:,i])
        xticks(range(0),attributeNames)
        title(attributeNames[i])
    suptitle("Boxplot of the attributes")
    show()

#Create histograms for the nine attributes
def histogram(X,attributeNames,y):
    attr = 9
    fig = figure()
    fig.subplots_adjust(hspace=.5)
    for i in range(attr):
        subplot(4,3,i+1)
        hist(X[:,i],bins=15)
        title(attributeNames[i])
    subplot(4,3,11)
    hist(y)
    title("CHD")
    suptitle("Histogram of the attributes")
    show()
    

#Plot the data
def plotTwoAttributes(attr1, attr2, X, y, classNames, attributeNames):
    C = len(classNames)
    f = figure()
    f.hold()
    s = attributeNames[attr1] + " vs. " + attributeNames[attr2]
    title(s)
    for c in range(C):
        class_mask = y.A.ravel()==c
        plot(X[class_mask,attr1], X[class_mask,attr2], 'o')
    legend([convertToWord(i) for i in classNames])
    xlabel(attributeNames[attr1])
    ylabel(attributeNames[attr2])
    show()


#Compute principal componentss
def computePrincipalComponents(X,s):
    Y = X - np.ones((len(X),1))*X.mean(0)
    
    U,S,V = linalg.svd(Y,full_matrices=False)
    
    rho = (S*S) / (S*S).sum()
    
    figure()
    plot(range(1,len(rho)+1),rho,'o-')
    title('Variance explained by principal components');
    xlabel('Principal component');
    ylabel('Variance explained');
    show()
    

#Plot principal components against each other
def plotPrincipalComponents(principal1, principal2, X, y, classNames):
    C = len(classNames)    
    Y = X - np.ones((len(X),1))*X.mean(0)
    
    U,S,V = linalg.svd(Y,full_matrices=False)
    V = mat(V).T
    
    Z = Y * V
    
    # Plot PCA of the data
    f = figure()
    f.hold()
    title('Data projected onto Principal Components')
    for c in range(C):
        class_mask = y.A.ravel()==c
        plot(Z[class_mask,principal1], Z[class_mask,principal2], 'o')
    legend([convertToWord(i) for i in classNames])
    xlabel('PC{0}'.format(principal1+1))
    ylabel('PC{0}'.format(principal2+1))
    show()
    
# Gets the direction of a certain principal component
def getPCADirections(X):
    Y = X - np.ones((len(X),1))*X.mean(0)
    U,S,V = linalg.svd(Y,full_matrices=False)
    
    print('Calculating direction of principal components: ')
    
    return V
    
    
#Calculate similarities
def calculateSim(row, similarity_measure, X):
    print("Id of interest: ",row)
    print(X[row])
    N, M = shape(X)
    noti = range(0,row) + range(row+1,N)
    sim = similarity(X[row,:], X[noti,:], similarity_measure)
    sim = sim.tolist()[0]
    sim_to_index = sorted(zip(sim,noti))
    
    for ms in range(3):
        im_id = sim_to_index[-ms-1][1]
        im_sim = sim_to_index[-ms-1][0]
        print("Id: ",im_id)
        print("Similarity: ", im_sim)
        print(X[im_id])
  
#Plot three attributes against each other      
def plot3D(X,y,classNames,attr1,attr2,attr3,attributeNames):
    f = figure()
    hold(True)
    colors = ['blue', 'green']
    ax = f.add_subplot(111, projection='3d')
    for c in range(C):
        class_mask = (y==c).A.ravel()
        ax.scatter(X[class_mask,attr1].A, X[class_mask,attr2].A, X[class_mask,attr3].A, c=colors[c])    
    ax.set_xlabel(attributeNames[attr1])
    ax.set_ylabel(attributeNames[attr2])
    ax.set_zlabel(attributeNames[attr3])
    legend(attributeNames)
    title("3D plot of attributes")
    show()
    
#Plot principal components in 3D-space
def plot3DPrincipalComponents(X,y,classNames,prin1,prin2,prin3,attributeNames):
    C = len(classNames)    
    Y = X - np.ones((len(X),1))*X.mean(0)
    
    U,S,V = linalg.svd(Y,full_matrices=False)
    V = mat(V).T
    
    Z = Y * V

    f = figure()
    hold(True)
    colors = ['blue', 'green']
    ax = f.add_subplot(111, projection='3d')
    for c in range(C):
        class_mask = (y==c).A.ravel()
        ax.scatter(Z[class_mask,prin1].A, Z[class_mask,prin2].A, Z[class_mask,prin3].A, c=colors[c])    
    ax.set_xlabel('PC{0}'.format(prin1+1))
    ax.set_ylabel('PC{0}'.format(prin2+1))
    ax.set_zlabel('PC{0}'.format(prin3+1))
    title("3D plot of principal components")
    legend(attributeNames)

        
#Using CHD as attribute
size = 463
positive = 160
noAttributes = 9

#Open data
doc = xlrd.open_workbook('pathtodataset/dataset.xls').sheet_by_index(0)

#Get Attributes
attributeNames = doc.row_values(0,1,noAttributes+1)

#Create classes
classLabels = doc.col_values(noAttributes+1,1,size)
classNames = sorted(set(classLabels))
classDict = dict(zip(classNames,range(2)))

y = np.mat([classDict[value] for value in classLabels]).T

yPositive = np.mat([[1] for i in range(positive)])
yNegative = np.mat([[0] for i in range(size-positive)])

N = len(y)
M = len(attributeNames)
C = len(classNames)


#Create matrix holding data
X = np.mat(np.empty((size-1,noAttributes)))
for i, col_id in enumerate(range(1,noAttributes+1)):
    if(attributeNames[col_id-1] == "famhist"):
        temp12 = [convert(i2) for i2 in doc.col_values(col_id,1,size)]
        X[:,i] = np.mat(temp12).T
    else:
        X[:,i] = np.mat(doc.col_values(col_id,1,size)).T
        
        
#Stanardize data       
XStandardized = zscore(X, ddof=1)

#Find all positive CHD
XPositive = X[y.A.ravel()==1,:]
XPositiveStd = zscore(XPositive,ddof=1)
XPositiveStd2 = XStandardized[y.A.ravel()==1,:]
#All negative CHD
XNegative = X[y.A.ravel()==0,:]
XNegativeStd = zscore(XNegative,ddof=1)

#Calcuate mean and variance
print("Calculate statistics")
calculateStatistics(X,noAttributes)

#Make histograms
histogram(X,attributeNames,y)

#Make boxplots
boxPlot(X,attributeNames)

#Plot alcohol and tobacco
plotTwoAttributes(1,7,X,y,classNames,attributeNames)
plotTwoAttributes(5,7,X,y,classNames,attributeNames)
plotTwoAttributes(3,6,X,y,classNames,attributeNames)

#Plot the variance explained by the principal components
computePrincipalComponents(XStandardized, "For both negative and positive CHD")

#Plot the data projected to the first two principal components
plotPrincipalComponents(0,1,XStandardized,y,classNames)

#Calculate directions of PCAs
print("For both positive and negative:")
print(getPCADirections(XStandardized))
        

#Calculate correlation between attributes
corrcoef = corrcoef(X.T,y.T)

print("How the attributes correlate:")
print(corrcoef)

plot3DPrincipalComponents(X,y,classNames,0,1,2,attributeNames)

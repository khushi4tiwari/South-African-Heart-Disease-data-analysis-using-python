
from sklearn.mixture import GMM
from pylab import *
from sklearn import cross_validation
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from subprocess import call
import re
import os
from toolbox_02450 import gausKernelDensity
from toolbox_02450 import gauss_2d
from sklearn.neighbors import NearestNeighbors

def max(x1,x2):
    if x1 > x2:
        return x1
    else:
        return x2
        
def min(x1,x2):
    if x1 < x2:
        return x1
    else:
        return x2


def gmm(X,y,M,C,K=4,cov_type='diag',reps=10):
    
    
    # Fit Gaussian mixture model
    
    gmm = GMM(n_components=K, covariance_type=cov_type, n_init=reps, params='wmc').fit(X)
    
    cls = gmm.predict(X)    # extract cluster labels
    
    cds = gmm.means_        # extract cluster centroids (means of gaussians)
    
    covs = gmm.covars_      # extract cluster shapes (covariances of gaussians)
    
    
    
    if cov_type == 'diag':
    
        new_covs = np.zeros([K,M,M])
    
        count = 0
    
        for elem in covs:
    
            temp_m = np.zeros([M,M])
    
            for i in range(len(elem)):
    
                temp_m[i][i] = elem[i]
    
            new_covs[count] = temp_m
    
            count += 1
    
        covs = new_covs
        
    clusterPlot(X,cls,K,C,y,cds,covs)
    
    

def clusterPlot(X,cls,K,C,y=None,centroids=None,covars=None):
    # Plot results:
    
    figure(figsize=(14,9))
    #X2 = X[:,0:2]
    #M=2
    
    #for c in range(C):
    #    class_mask = y.A.ravel()==c
    #    plot(X[class_mask,attr1], X[class_mask,attr2], 'o')
    ncolors = K+2
    y2 = np.asarray(y)
    

    hold(True)
    wrong = 0
    correct = 0
    clusters = 0
    colors = [0]*ncolors
    for color in range(ncolors):
        colors[color] = cm.jet.__call__(color*255/(ncolors-1))[:3]
    for i,cs in enumerate(np.unique(y2)):
        plot(X[(y2==cs).ravel(),0], X[(y2==cs).ravel(),1], 'o', markeredgecolor='k', markerfacecolor=colors[i],markersize=6, zorder=2)
    for i,cr in enumerate(np.unique(cls)):
        clusters += 1
        plot(X[(cls==cr).ravel(),0], X[(cls==cr).ravel(),1], 'o', markersize=12, markeredgecolor=colors[i+2], markerfacecolor='None', markeredgewidth=3, zorder=1)
        Z = X[(cls==cr).ravel(),:]
        print i
        print "Size of cluster: " + str(len(Z))
        
        y3 = y2[cls==cr.ravel()]
        Z1 = None
        Z2 = None
        for i,cs in enumerate(np.unique(y3)):
            #print i
            if i == 0:
                Z1 = Z[(y3==cs).ravel(),:]
            else:
                Z2 = Z[(y3==cs).ravel(),:]
        if (Z1 is None):
            negs = 0
        else:
            negs = len(Z1)
        if (Z2 is None):
            poss = 0
        else:
            poss = len(Z2)
        correct += max(negs,poss)
        wrong += min(negs,poss)
        print "Negatives: " + str(negs)
        print "Positives: " + str(poss)
        if negs == max(negs,poss):
            print "Classified as negatives"
        else:
            print "Classified as possitives"
    

    if centroids!=None:
        for cd in range(centroids.shape[0]):
            plot(centroids[cd,0], centroids[cd,1], '*', markersize=22, markeredgecolor='k', markerfacecolor=colors[cd+2], markeredgewidth=2, zorder=3)
            
    if covars!=None:
        centroids2 = centroids[:,0:2]
        for cd in range(centroids2.shape[0]):
            x1, x2 = gauss_2d(centroids2[cd],covars[cd,0:2,0:2])
            plot(x1,x2,'-', color=colors[cd+2], linewidth=3, zorder=5)
    hold(False)
            
    legend_items = ["Negative","Positive"]
    for i in range(clusters):
        legend_items.append(i)
    for i in range(clusters):
        legend_items.append(i)
    #legend_items = np.unique(y2).tolist()+np.unique(cls).tolist()+np.unique(cls).tolist()
    for i in range(len(legend_items)):
        if i<C: legend_items[i] = 'Class: {0}'.format(legend_items[i]);
        elif i<C+clusters: legend_items[i] = 'Cluster: {0}'.format(legend_items[i]);
        else: legend_items[i] = 'Centroid: {0}'.format(legend_items[i]);
    legend(legend_items, numpoints=1, markerscale=.75, prop={'size': 9})
    xlabel("PC1")
    ylabel("PC2")
    title("Plot of clusters")


    #clusterplot(X2, clusterid=cls, centroids=cds, y=y, covars=covs)

    print "Misclassification rate:"
    print (double(wrong)/double(wrong + correct))

    
    show()


def getPrincipalComponents(X):
    Y = X - np.ones((len(X),1))*X.mean(0)
    
    U,S,V = linalg.svd(Y,full_matrices=False)
    
    return U
    
def CVK(X,KRange,covar_type='diag',reps=10):
    N, M = X.shape
    T = len(KRange)
    
    CVE = np.zeros((T,1))
    
    # K-fold crossvalidation
    CV = cross_validation.KFold(N,5,shuffle=True)

    for t,K in enumerate(KRange):
            print('Fitting model for K={0}\n'.format(K))
    
            # Fit Gaussian mixture model
            gmm = GMM(n_components=K, covariance_type=covar_type, n_init=reps, params='wmc').fit(X)
            
            # For each crossvalidation fold
            for train_index, test_index in CV:
    
                # extract training and test set for current CV fold
                X_train = X[train_index]
                X_test = X[test_index]
    
                # Fit Gaussian mixture model to X_train
                gmm = GMM(n_components=K, covariance_type=covar_type, n_init=reps, params='wmc').fit(X_train)
    
                # compute negative log likelihood of X_test
                CVE[t] += -gmm.score(X_test).sum()
                #print CVE[t]
                
        # Plot results
    return CVE
    
    #figure(); hold(True)
    #plot(KRange, 2*CVE)
    #legend(['Crossvalidation'])
    #xlabel('K')
    #show()
    
def plotCVK(X,KRange,iterations=1,covar_type='diag',reps=10):
    figure()
    hold(True)
    legend_items = []
    for i in range(iterations):
        plot(KRange, 2*CVK(X,KRange,covar_type,reps))
        s = 'Cross Validation ' + str(i)        
        legend_items.append(s)
    legend(legend_items)
    xlabel('k')
    title('Cross Validation by Gaussian Mixture Modelling')
    show()
    
def hierarchicalClustering(X,y,Maxclust, C, Method = 'single', Metric = 'euclidean'):
    # Perform hierarchical/agglomerative clustering on data matrix
    
    Z = linkage(X, method=Method, metric=Metric)
    
    # Compute and display clusters by thresholding the dendrogram
    cls = fcluster(Z, criterion='maxclust', t=Maxclust)
    figure()
    #clusterplot(X, cls.reshape(cls.shape[0],1), y=y)
    clusterPlot(X, cls.reshape(cls.shape[0],1), Maxclust, C, y=y)

    # Display dendrogram
    max_display_levels=7
    figure()
    dendrogram(Z, truncate_mode='level', p=max_display_levels, color_threshold=0.5*np.max(Z[:,2]))
    title("Dendrgram of the Hierarchical Clustering")
    show()
    
def getColor(k,colors):
    colors[k]
    
def convertToBinary(X,Y=None):
    ''' Force binary representation of the matrix, according to X>median(X) '''
    if Y==None:
        X = np.matrix(X)
        Xmedians = np.ones((np.shape(X)[0],1)) * np.median(X,0)
        Xflags = X>Xmedians
        X[Xflags] = 1; X[~Xflags] = 0
        return X
    else:
        X = np.matrix(X); Y = np.matrix(Y);
        XYmedian= np.median(np.bmat('X; Y'),0)
        Xmedians = np.ones((np.shape(X)[0],1)) * XYmedian
        Xflags = X>Xmedians
        X[Xflags] = 1; X[~Xflags] = 0
        Ymedians = np.ones((np.shape(Y)[0],1)) * XYmedian
        Yflags = Y>Ymedians
        Y[Yflags] = 1; Y[~Yflags] = 0
        return [X,Y]
        

def doApriori(filename, minSup = 80, minConf = 100, maxRule = 4):    
    # Run Apriori Algorithm
    print('Mining for frequent itemsets by the Apriori algorithm')
    status1 = call('apriori.exe -f"," -s{0} -v"[Sup. %0S]" {1} apriori_temp1.txt'.format(minSup, filename))
    if status1!=0:
        print('An error occured while calling apriori, a likely cause is that minSup was set to high such that no frequent itemsets were generated or spaces are included in the path to the apriori files.')
        exit()
    if minConf>0:
        print('Mining for associations by the Apriori algorithm')
        status2 = call('apriori.exe -tr -f"," -n{0} -c{1} -s{2} -v"[Conf. %0C,Sup. %0S]" {3} apriori_temp2.txt'.format(maxRule, minConf, minSup, filename))
        if status2!=0:
            print('An error occured while calling apriori')
            exit()
    print('Apriori analysis done, extracting results')
    
    
    # Extract information from stored files apriori_temp1.txt and apriori_temp2.txt
    f = open('apriori_temp1.txt','r')
    lines = f.readlines()
    f.close()
    # Extract Frequent Itemsets
    FrequentItemsets = ['']*len(lines)
    sup = np.zeros((len(lines),1))
    for i,line in enumerate(lines):
        FrequentItemsets[i] = line[0:-1]
        sup[i] = re.findall(' \d*]', line)[0][1:-1]
    os.remove('apriori_temp1.txt')
        
    # Read the file
    f = open('apriori_temp2.txt','r')
    lines = f.readlines()
    f.close()
    # Extract Association rules
    AssocRules = ['']*len(lines)
    conf = np.zeros((len(lines),1))
    for i,line in enumerate(lines):
        AssocRules[i] = line[0:-1]
        conf[i] = re.findall(' \d*,', line)[0][1:-1]
    os.remove('apriori_temp2.txt')    
    
    # sort (FrequentItemsets by support value, AssocRules by confidence value)
    AssocRulesSorted = [AssocRules[item] for item in np.argsort(conf,axis=0).ravel()]
    AssocRulesSorted.reverse()
    FrequentItemsetsSorted = [FrequentItemsets[item] for item in np.argsort(sup,axis=0).ravel()]
    FrequentItemsetsSorted.reverse()
        
    # Print the results
    import time; time.sleep(.5)    
    print('\n')
    print('RESULTS:\n')
    print('Frequent itemsets:')
    for i,item in enumerate(FrequentItemsetsSorted):
        print('Item: {0}'.format(item))
    print('\n')
    print('Association rules:')
    for i,item in enumerate(AssocRulesSorted):
        print('Rule: {0}'.format(item))
    
    
def outlierDetection(X, objects = 20):
    

    ### Gausian Kernel density estimator
    # cross-validate kernel width by leave-one-out-cross-validation
    # (efficient implementation in gausKernelDensity function)
    # evaluate for range of kernel widths
    widths = X.var(axis=0).max() * (2.0**np.arange(-10,3))
    logP = np.zeros(np.size(widths))
    for i,w in enumerate(widths):
       density, log_density = gausKernelDensity(X,w)
       logP[i] = log_density.sum()
    val = logP.max()
    ind = logP.argmax()
    
    width=widths[ind]
    print('Optimal estimated width is: {0}'.format(width))
    
    # evaluate density for estimated width
    density, log_density = gausKernelDensity(X,width)
    
    # Sort the densities
    i = (density.argsort(axis=0)).ravel()
    density = density[i]
    
    # Plot density estimate of outlier score
    figure()
    bar(range(objects),density[:objects])
    title('Density estimate')
    
    # Plot possible outliers
    #figure()
    print "For Gaussian Kernel Density"
    for k in range(1,objects + 1):
        print i[k]
        #subplot(4,5,k)
        #imshow(np.reshape(X[i[k],:], (1,9)).T, cmap=cm.binary)
        #xticks([]); yticks([])
        #if k==3: title('Gaussian Kernel Density: Possible outliers')
    
    
    
    ### K-neighbors density estimator
    # Neighbor to use:
    K = 5
    
    # Find the k nearest neighbors
    knn = NearestNeighbors(n_neighbors=K).fit(X)
    D, i = knn.kneighbors(X)
    
    density = 1./(D.sum(axis=1)/K)
    
    # Sort the scores
    i = density.argsort()
    density = density[i]
    
    # Plot k-neighbor estimate of outlier score (distances)
    figure()
    bar(range(objects),density[:objects])
    title('KNN density: Outlier score')
    # Plot possible outliers
    #figure()
    print "\n"
    print "For KNN density"
    for k in range(1,objects + 1):
        print i[k]        
        #subplot(4,5,k)
        #imshow(np.reshape(X[i[k],:], (1,9)).T, cmap=cm.binary)
        #xticks([]); yticks([])
        #if k==3: title('KNN density: Possible outliers')
    
    
    
    ### K-nearest neigbor average relative density
    # Compute the average relative density
    
    knn = NearestNeighbors(n_neighbors=K).fit(X)
    D, i = knn.kneighbors(X)
    density = 1./(D.sum(axis=1)/K)
    avg_rel_density = density/(density[i[:,1:]].sum(axis=1)/K)
    
    # Sort the avg.rel.densities
    i_avg_rel = avg_rel_density.argsort()
    avg_rel_density = avg_rel_density[i_avg_rel]
    
    # Plot k-neighbor estimate of outlier score (distances)
    figure()
    bar(range(objects),avg_rel_density[:objects])
    title('KNN average relative density: Outlier score')
    # Plot possible outliers
    #figure()
    print "\n"
    print "For KNN average relative density"
    for k in range(1,objects + 1):
        print i_avg_rel[k]        
        #subplot(4,5,k)
        #imshow(np.reshape(X[i_avg_rel[k],:], (1,9)).T, cmap=cm.binary)
        #xticks([]); yticks([])
        #if k==3: title('KNN average relative density: Possible outliers')
    
    
    
    ### Distance to 5'th nearest neighbor outlier score
    K = 25
    
    # Find the k nearest neighbors
    knn = NearestNeighbors(n_neighbors=K).fit(X)
    D, i = knn.kneighbors(X)
    
    # Outlier score
    score = D[:,K-1]
    # Sort the scores
    i = score.argsort()
    score = score[i[::-1]]
    
    # Plot k-neighbor estimate of outlier score (distances)
    figure()
    bar(range(objects),score[:objects])
    title('25th neighbor distance: Outlier score')
    # Plot possible outliers
    #figure()
    print "\n"
    print "For 5'th neighbour distance"
    for k in range(1,objects + 1):
        print i[462-k]        
        #subplot(4,5,k)
        #imshow(np.reshape(X[i[k],:], (1,9)).T, cmap=cm.binary); 
        #xticks([]); yticks([])
        #if k==3: title('5th neighbor distance: Possible outliers')
    
    
    
    # Plot random digits (the first 20 in the data set), for comparison
    #figure()
    #for k in range(1,objects + 1):
    #    subplot(4,5,k);
    #    imshow(np.reshape(X[k,:], (1,9)).T, cmap=cm.binary); 
    #    xticks([]); yticks([])
    #    if k==3: title('Random digits from data set')    
    show()
    


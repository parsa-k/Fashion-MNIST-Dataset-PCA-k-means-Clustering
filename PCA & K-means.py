
"""
Created on Thu Jan  1 14:57:28 2021
Fashion MNIST Dataset, Using of PCA & K-means Clustering
@author: parsa khorrami
"""

import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.decomposition import PCA
import plotly as py
import plotly.graph_objs as go
import plotly.express as px

#Load train data set
train = pd.read_csv("fashion-mnist_train.csv")
#Load test data set
test = pd.read_csv("fashion-mnist_test.csv")
print(f'train dataset has: {train.shape[0]} rows {train.shape[1]} columns')
print(f'test dataset has : {test.shape]0[} rows {test.shape]1[} columns')


# Preprocess the data
height = 28
width = 28
# defining perprocces function
def preprocess_data(data):
    #Label(target)
    y_output = data.label
    #Features(image)
    x_output = np.array(data.values[:,1:])
    
    #Normalize data
    x_output = x_output.reshape(data.shape[0], height, width)
    #Padding the images by 2 pixels since in the paper input images were 32x32
    return x_output, y_output

X_train, Y_train = preprocess_data(train)
#Reshapeing X to a 2D array for PCA and then k-means
X = X_train.reshape(-1,X_train.shape[1]*X_train.shape[2]) #We will only be using X for clustering
y = Y_train
#Sanity check
print ("The shape of X is " + str(X.shape))
print ("The shape of y is " + str(y.shape)) #We will be using y only to check our clustering


#Visualise an image 
n= 2 #Enter Index here to View the image 
plt.imshow(X[n].reshape(X_train.shape[1], X_train.shape[2]), cmap = plt.cm.binary)
plt.show()

#PCA (Principle Component Analysis)
# To perform PCA we must first change the mean to 0 and variance to 1 for X using StandardScalar
Clus_dataSet = StandardScaler().fit_transform(X) #(mean = 0 and variance = 1)

# Make an instance of the Model
variance = 0.98 #The higher the explained variance the more accurate the model will remain
pca = PCA(variance)

#fit the data according to our PCA instance
pca.fit(Clus_dataSet)
PCA(copy=True, iterated_power='auto', n_components=0.98, random_state=None,
    svd_solver='auto', tol=0.0, whiten=False)

print("Number of components before PCA  = " + str(X.shape[1]))
print("Number of components after PCA 0.98 = " + str(pca.n_components_)) #dimension reduced from 784

#Transform our data according to our PCA instance
Clus_dataSet = pca.transform(Clus_dataSet)
print("Dimension of our data after PCA  = " + str(Clus_dataSet.shape))

#To visualise the data inversed from PCA
approximation = pca.inverse_transform(Clus_dataSet)
print("Dimension of our data after inverse transforming the PCA  = " + str(approximation.shape))

#image reconstruction using the less dimensioned data
plt.figure(figsize=(8,4));

n = 2 #index value, change to view different data
# Original Image
plt.subplot(1, 2, 1);
plt.imshow(X[n].reshape(X_train.shape[1], X_train.shape[2]),
              cmap = plt.cm.gray,);
plt.xlabel(str(X.shape[1])+' components', fontsize = 14)
plt.title('Original Image', fontsize = 20);

# 196 principal components
plt.subplot(1, 2, 2);
plt.imshow(approximation[n].reshape(X_train.shape[1], X_train.shape[2]),
              cmap = plt.cm.gray,);
plt.xlabel(str(Clus_dataSet.shape[1]) +' components', fontsize = 14)
plt.title(str(variance * 100) + '% of Variance Retained', fontsize = 20);
plt.show()



#K-MEANS ++
#n_clusters = 10 because INDEX has 10 values. Not the best value but a simple logic.
#The value of n_init at 35 yields good results so we will use it. For confirmation us the above code.
k_means = KMeans(init = "k-means++", n_clusters = 10, n_init = 35)
#fit the data to our k_means model
k_means.fit(Clus_dataSet)
KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
       n_clusters=10, n_init=35, n_jobs=None, precompute_distances='auto',
       random_state=None, tol=0.0001, verbose=0)

k_means_labels = k_means.labels_ #List of labels of each dataset
print("The list of labels of the clusters are " + str(np.unique(k_means_labels)))

G = len(np.unique(k_means_labels)) #Number of labels

#2D matrix  for an array of indexes of the given label
cluster_index= [[] for i in range(G)]
for i, label in enumerate(k_means_labels,0):
    for n in range(G):
        if label == n:
            cluster_index[n].append(i)
        else:
            continue

#Visualisation for clusters = clust
plt.figure(figsize=(20,20));
clust = 8 #enter label number to visualise
num = 100 #num of data to visualize from the cluster
for i in range(1,num): 
    plt.subplot(10, 10, i); #(Number of rows, Number of column per row, item number)
    plt.imshow(X[cluster_index[clust][i+500]].reshape(X_train.shape[1], X_train.shape[2]), cmap = plt.cm.binary);
    
plt.show()



# ploting result
Y_clust = [[] for i in range(G)]

for n in range(G):
    Y_clust[n] = y[cluster_index[n]] 
#Y_clust[0] contains array of "correct" category from y_train for the cluster_index[0]
    assert(len(Y_clust[n]) == len(cluster_index[n])) #dimension confirmation

#counts the number of each category in each cluster
def counter(cluster):
    unique, counts = np.unique(cluster, return_counts=True)
    label_index = dict(zip(unique, counts))
    return label_index

label_count= [[] for i in range(G)]
for n in range(G):
    label_count[n] = counter(Y_clust[n])

print(f'Number of items of a certain category in cluster 1 is: {label_count[1]}')


class_names = {0:'T-shirt/top', 1:'Trouser',2: 'Pullover',3: 'Dress',4: 'Coat',5:
               'Sandal',6: 'Shirt', 7:'Sneaker',8:  'Bag',9: 'Ankle boot'} #Dictionary of class names


#A function to plot a bar graph for visualising the number of items of certain category in a cluster
def plotter(label_dict):
    plt.bar(range(len(label_dict)), list(label_dict.values()), align='center')
    a = []
    for i in [*label_dict]: a.append(class_names[i])
    plt.xticks(range(len(label_dict)), list(a), rotation=45, rotation_mode='anchor')


#Bar graph with the number of items of different categories clustered in it
plt.figure(figsize=(20,20))
for i in range (1,11):
    plt.subplot(5, 2, i)
    plotter(label_count[i-1]) 
    plt.title("Cluster" + str(i-1))


plt.show()
k_means_cluster_centers = k_means.cluster_centers_ #numpy array of cluster centers


print("(clusters,features)"+ str(k_means_cluster_centers.shape)) #comes from 10 clusters and 420 features


#cluster visualisation
my_members = (k_means_labels == 3) 
#Enter different Cluster number to view its 3D plot

my_members.shape
fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(1,1,1,projection='3d')


#Clus_dataSet.shape
#Clus_dataSet[my_members,300].shape
ax.plot(Clus_dataSet[my_members, 0],
 Clus_dataSe [my_members,1], 
Clus_dataSet[my_members,2] ,
 'w', markerfacecolor="blue", marker='.',markersize=10)


#3D Plotly Visualisation of Clusters using go

layout = go.Layout(
    title='<b>Cluster Visualisation</b>',
    yaxis=dict(
        title='<i>Y</i>'
    ),
    xaxis=dict(
        title='<i>X</i>'
    )
)

colors = ['red','green' ,'blue','purple','magenta','yellow','cyan','maroon','teal','black']
trace = [ go.Scatter3d() for _ in range(11)]
for i in range(0,10):
    my_members = (k_means_labels == i)
    index = [h for h, g in enumerate(my_members) if g]
    trace[i] = go.Scatter3d(
            x=Clus_dataSet[my_members, 0],
            y=Clus_dataSet[my_members, 1],
            z=Clus_dataSet[my_members, 2],
            mode='markers',
            marker = dict(size = 2,color = colors[i]),
            hovertext=index,
            name='Cluster'+str(i),
   
            )
fig = go.Figure(data=[trace[0],trace[1],trace[2],trace[3],trace[4],trace[5],trace[6],trace[7],trace[8],trace[9]], layout=layout)
py.offline.iplot(fig)


import numpy as np                          # For mathematical computation on matrices
import matplotlib.pyplot as plt             # To plot the resulting clusters
from sklearn.cluster import KMeans          # To fit KMeans to data
from sklearn.decomposition import PCA       # To apply PCA to data
from prettytable import PrettyTable         # To print cluster and index/rows assigned to that cluster


tt = PrettyTable(['Cluster no.','Index no.'])                               # Define Table for displaying cLuster number with the index number(row no. of data)
np.random.seed(42)                                                          # Initiaize see with a random number

data = np.genfromtxt('Data.csv', dtype=float, delimiter=',')      # Read the data from the csv file using ',' as the separator for features
index = range(0, len(data))                                                 # Define the index (no of rows) of data

n_samples, n_features = data.shape                                          # Get the number of samples(no of rows) and no of features (no of columns) using shape of data
n_clusters = 10                                                             # Define the number of clusters we want


print("n_clusters: %d, \t n_samples %d, \t n_features %d" % (n_clusters, n_samples, n_features))    #Printing values


def bench_k_means(estimator, name, data):                                   # Define function to handle k means
    estimator.fit_predict(data)                                             # Fit the model on data
    ans = estimator.predict(data)                                           # Predict using the data
    zipped = zip(index,ans)                                                 # Zip index(row no) with its predicted cluster
    zipped.sort(key=lambda t: t[1])                                         # Sort the pairs(index, predicted cluster) according to the cluster no predicted

    for a, b in zipped:                                                     # Display the index no and the predicted cluster with feature values of each row
        tt.add_row([b , a])


bench_k_means(KMeans(init='k-means++', n_clusters=n_clusters, n_init=10), name="k-means++", data=data)      # Call the function to perform KMeans and print the table of results

print tt                                            #print the table with cluster no and row no./index

# ------------------------------------  DISPLAYING THE CLUSTERS ON A 2D FIGURE  ------------------------------------

# Displaying the clusters formed, since we have more than 3 features, we will have to use PCA to reduce the dimensions from 'n_features' to '2' to display the clusters
reduced_data = PCA(n_components=2).fit_transform(data)                  # Apply PCA to data
kmeans = KMeans(init='k-means++', n_clusters=n_clusters, n_init=10)     # Define KMeans model to be used
kmeans.fit(reduced_data)                                                # Apply KMeans to the reduced dimension data

# Step size of the mesh. Decrease to increase the quality of the VQ
h = .02                                                                 # point in the mesh [x_min, x_max]x[y_min, y_max]

# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1           # Define the limits for the x-axis using the min and max value of 'x' in data
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1           # Define the limits for the y-axis using the min and max value of 'y' in data
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))        # Define the points for color plot mesh using limits and stepsize 'h' ; The purpose of meshgrid is to create a rectangular grid out of an array of x values and an array of y values
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])                                   # Obtain labels for each point in mesh. Use last trained model; This will help us create the image with different colors for the clusters since we are predicting cluster for each point in meshgrid

# Put the result into a color plot
Z = Z.reshape(xx.shape)                 # Reshape Z into shape of 'xx' - you can think of Z as an image with areas of different colors for each of the clusters
plt.figure(1)                           # Create a new figure
plt.clf()                               # Clear the figure

plt.imshow(Z, interpolation='nearest',extent=(xx.min(), xx.max(), yy.min(), yy.max()), cmap=plt.cm.Paired, aspect='auto', origin='lower')       # Using imshow() to display RGB image on the figure

plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)                                                # Plot the data(reduced dimensions) on the figure
centroids = kmeans.cluster_centers_                                                                                 # Get Centres of the clusters
plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=169, linewidths=3, color='w', zorder=10)                # Scatter centroid points - Plot the centroids as a white X
plt.title('K-means clustering on the dataset (PCA-reduced data)\n' 'Centroids are marked with white cross')   # Add title to the plot figure

plt.xlim(x_min, x_max)                  # Set the x axis scale limit to xmin, xmax
plt.ylim(y_min, y_max)                  # Set the y axis scale limit to ymin, ymax
plt.xticks(())                          # Display xticks on s-axis
plt.yticks(())                          # Display yticks on y-axis
plt.show()                              # Show the plot
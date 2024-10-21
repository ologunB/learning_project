
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Sample customer data for clustering
customer_data = np.array([[15, 39], [16, 81], [17, 6], [18, 94], [19, 3], [20, 72], 
                          [21, 12], [22, 67], [23, 34], [24, 83], [25, 44]])

# Apply K-Means clustering algorithm
kmeans = KMeans(n_clusters=3, random_state=0).fit(customer_data)

# Plot clusters
plt.scatter(customer_data[:, 0], customer_data[:, 1], c=kmeans.labels_, cmap='rainbow')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color='black', marker='x', s=100)
plt.title("Customer Segmentation")
plt.xlabel("Spending on Item 1")
plt.ylabel("Spending on Item 2")
plt.show()

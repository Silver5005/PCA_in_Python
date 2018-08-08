# Use numpy genfromtxt() to import CSV data as matrix. (10 stocks x 504 days)
import numpy as np # Numpy for matrix math
import pandas as pd

# First we import our stock data.  We will reduce 10 unique stocks to a dimensionality of 2 using PCA to allow for visualization.
# Each col = unique stock, in this order: AAPL, AMZN, MSFT, GOOG, FB, BA, UNH, GS, MMM, HD
# Each value in matrix represents a % change for that day. (decimalized)
my_data = pd.read_csv('PCAstocks.csv')

# Now we perform PCA on these 10 stocks to reduce dimensionality to 2.
from sklearn.decomposition import PCA
#num_pc = 10 # Optional input to show amount of variability explained by each unique input
num_pc = 2 # Actual desired input

X = my_data
[n,m] = X.shape
print ('The number of timestamps is {}.'.format(n)) # Prints number of days (rows)
print ('The number of stocks is {}.'.format(m)) # Prints number of stocks (cols)

# Fit our data
pca = PCA(n_components=num_pc) # number of principal components
pca.fit(X)

percentage =  pca.explained_variance_ratio_ # Variation of data explained with first 2 principal componets
percentage_cum = np.cumsum(percentage) # Cumulative sum
print('{0:.2f}% of the variance is explained by the first 2 PCs'.format(percentage_cum[-1]*100)) # Check value with print
# First 2 PC's only explain ~61% of this given dataset, definitely not the best representation but we'll stick with it for now.

pca_components = pca.components_

import matplotlib.pyplot as plt # Matplotlib for visualization
x = np.arange(1,len(percentage)+1,1)

# Prints bar chart showing % of variation explained by each PC.
plt.subplot(1, 2, 1)
plt.bar(x, percentage*100, align = "center")
plt.title('Contribution of principal components',fontsize = 16)
plt.xlabel('principal components',fontsize = 16)
plt.ylabel('percentage',fontsize = 16)
plt.xticks(x,fontsize = 16)
plt.yticks(fontsize = 16)
plt.xlim([0, num_pc+1])

# Same concept, but a line graph over time.
plt.subplot(1, 2, 2)
plt.plot(x, percentage_cum*100,'ro-')
plt.xlabel('principal components',fontsize = 16)
plt.ylabel('percentage',fontsize = 16)
plt.title('Cumulative contribution of principal components',fontsize = 16)
plt.xticks(x,fontsize = 16)
plt.yticks(fontsize = 16)
plt.xlim([1, num_pc])
plt.ylim([50,100])

factor_returns = X.dot(pca_components.T) # Dot-product of the transpose
# factor_returns = pd.DataFrame(columns=["0", "1"], # Create pandas dataframe
#                               index=my_data.index,
#                               data=factor_returns)
# factor_returns.head()


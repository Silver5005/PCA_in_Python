# PCA_in_Python
Principal Component Analysis on a data set of 10 stocks in order to reduce dimensionality to 2 principal components.  Numpy, Pandas.

Today we will perform PCA on a set of 10 stocks I've chosen, the top 5 holdings for the Nasdaq 
and top 5 holdings of the Dow Jonesm as of August 2018.

We first decide the number of dimensions we want to reduce our data set to.  We can do this by naively guessing and checking, 
or we can write a loop and see where the diminshing returns of adding dimensions (or principal componets) kicks in.

Here is what our graph looks like:

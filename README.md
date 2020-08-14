# PCA_in_Python
Principal Component Analysis on a data set of 10 stocks in order to reduce dimensionality to 2 principal components.  Numpy, Pandas.

Today we will perform PCA on a set of 10 stocks I've chosen, the top 5 holdings for the Nasdaq 
and top 5 holdings of the Dow Jones as of August 2018.   The data is included in the repository as 'PCAstocks.csv'.

Nasdaq: Apple, Amazon, Microsoft, Google, Facebook

Dow: Boing, United Health Group, Goldmansachs, 3M, Home Depot

We first decide the number of dimensions we want to reduce our data set to.  We can do this by naively guessing and checking, 
or we can write a loop and see where the diminshing returns of adding dimensions (or principal components) kicks in.

Here is what our graph looks like:
![figure_1](https://user-images.githubusercontent.com/34739163/43812928-d49103fa-9a80-11e8-87db-4e0391e8e699.png)

As we can see the effect of adding principal components drops off signficantly after the second (from ~15% to below 8%)
With this in mind we will make our desired dimensions 2, which is also nice because this allows for easy visualization.

We fit our data using the number of dimensions we want and scikitlearn's PCA().
Print out of the desired dimensions and variability of the data they explain:
![figure_2](https://user-images.githubusercontent.com/34739163/43813586-8868327a-9a83-11e8-840e-c22ccf539889.png)

Now we calculate our factor returns and factor exposures (risk variable).   We can then plot the factor exposure or our first principal
component against our second.  Here's the result:
![figure_3](https://user-images.githubusercontent.com/34739163/43813588-89d92420-9a83-11e8-858a-79e9276ec2e6.png)

Very interesting!  We can see two rather distinct groupings of the data. Non-coincidentally they group into their respective 
tech and or DOW industries, which is where the data was pulled. (top 5 holdings of QQQ, top 5 of DIA.)

PCA allows for us to reduce the number of dimensions we are working with, which when dealing with financial data can often
be extremely bountiful (curse of dimensionality).   This allows for easier visualization, computation, and intuition.  

We could further pair this with K means clustering as a way to cluster and classify our factor exposures, 
and we could try to classify any new data as one risk exposure or the other. 

PCA is one of the most common dimensionality reduction techniques in data science and rather easy to implement.

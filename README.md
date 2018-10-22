# My Data Science Projects

This is my collection of data science projects.
Each project has its own folder. 
More information about these projects is listed below.

## cryptocurrencies

Used in my blog post ["An Analysis of Cryptocurrency Trends"](http://johncrattz.com/an-analysis-of-cryptocurrency-trends/).

This is my first project.
This project analyzed cryptocurrency trends and attempted
to predict future values of cryptocurrencies based on their
previous values by training recurrent neural networks.
Predicting future values for financial assets based purely
on their previous values generally will not create very accurate
models, especially in a highly volatile market like the cryptocurrency
market in the latter half of 2017 through the beginning of 2018.
Still, it showcases my understanding of various Python data analysis,
data visualization, and machine learning libraries.

For Python, I used the NumPy, pandas, seaborn, Matplotlib, scikit-learn, and Keras libraries.

There is daily data [from Kaggle](https://www.kaggle.com/sudalairajkumar/cryptocurrencypricehistory)
and hourly data from [CryptoDataDownload.com](http://www.cryptodatadownload.com/)
stored in the **data** subfolder.

The primary file of interest is **Cryptocurrencies.py**.
The machine learning models are stored in the **models** subfolder.
The figures are stored in the **figures** subfolder.
Figures directly in that subfolder contain multiple plots per figure.
Figures containing plots of one cryptocurrency each are in corresponding 
subfolders, like **figures/Bitcoin**. 
The significance of figures is easily deduced from their names.
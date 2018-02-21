# My Data Science Projects

This is my collection of data science projects.
Each project has its own folder. 
More information about these projects is listed below.

## cryptocurrencies

Used in my blog post ["An Analysis of Cryptocurrency Trends"](http://johncrattz.com/an-analysis-of-cryptocurrency-trends/).

This was my first project. 
This project analyzed cryptocurrency trends and attempted
to predict future values of the currencies based on their 
previous values, but the predictions from the machine learning 
models did not closely follow the real values for dates after 
the end of the data. Predicting future values for financial assets 
based purely on their previous values generally will not
create very accurate models, especially in a highly volatile
market like the cryptocurrency market in the latter half of 2017 
through the beginning of 2018. Still, it showcases my understanding 
of various Python and R data analysis and data visualization libraries. 

For Python, I used the NumPy, pandas, seaborn, Matplotlib, Plotly, and Cufflinks libraries. 

For R, I used the ggplot2 and caret libraries, among several other smaller libraries and the builtin library.

The data is [from Kaggle](https://www.kaggle.com/sudalairajkumar/cryptocurrencypricehistory)
and is stored in the **data** subfolder.

### Python

Check out the IPython notebook (**Cryptocurrencies_v2.ipynb**) 
in the **python** subdirectory for a nice Plotly interactive plot of the value 
histories ("value trends" in the notebook) for the various cryptocurrencies.

The machine learning models that would be trained if the commented block 
of code above the "Model Loading" section in **Cryptocurrencies_v2.py** 
was uncommented and run are stored in the **models** subfolder. 
If **Cryptocurrencies_v2.py** is run, figures are stored in the **figures**
subfolder. Figures directly in that subfolder contain multiple plots per figure. 
Figures containing plots of one cryptocurrency each are in corresponding 
subfolders, like **figures/Bitcoin**. 
The significance of figures is easily deduced from their names.

### R

Check out the R file (**Cryptocurrencies.R**) in the **R** subdirectory.
I recommend running with RStudio - I have not run the script from any other IDE,
including the default R IDE.

The R script outputs the same files in the same subdirectories as the 
Python script. The figures directly in the **figures** subfolder which contain
multiple plots per figure are incorrect - repeating the corresponding Ripple figure
for each cell. I am still unsure of why this happens. Combining plots in ggplot2
has proven to be significantly more complicated than in matplotlib.
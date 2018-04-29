library(RMySQL)
library(ggplot2)
library(grid)
library(gridExtra)
library(reshape2)
library(scales)
library(quantmod)
library(plyr)

models_dir = 'models'
figures_dir = 'figures'
figure_dpi = 200

# Read from CSV files in the `data` subdirectory or from my mySQL
# server. The latter only works when running on my compter.
READ_FROM_CSV = TRUE

cryptocurrency_table_names = 
  c('bitcoin_cash', 'bitcoin', 'bitconnect', 'dash', 'ethereum_classic',
    'ethereum', 'iota', 'litecoin', 'monero', 'nem', 'neo', 'numeraire',
    'omisego', 'qtum', 'ripple', 'stratis', 'waves')
for (cryptocurrency_table_name in cryptocurrency_table_names) {
  assign(cryptocurrency_table_name, NULL)
}
if (!READ_FROM_CSV) {
  # Data Importing (SQL)
  con = dbConnect(MySQL(), user='john', 
                  password='secret', host='localhost',
                  dbname='cryptocurrencies')
  for (cryptocurrency_table_name in cryptocurrency_table_names) {
    assign(cryptocurrency_table_name, 
           dbReadTable(conn=con, name=cryptocurrency_table_name))
  }
} else {
  # Data Importing (CSV)
  for (cryptocurrency_table_name in cryptocurrency_table_names) {
    assign(cryptocurrency_table_name,
           read.csv(sprintf('../data/daily/%s_price.csv', cryptocurrency_table_name)))
  }
}
# Check the contents of the tables.
# for (cryptocurrency_table_name in cryptocurrency_table_names) {
#   cryptocurrency_table = lapply(cryptocurrency_table_name, get)[[1]]
#   print(sprintf('head(%s,5):', cryptocurrency_table_name))
#   print(head(cryptocurrency_table,5))
# }

currencies_labels_tickers = 
  matrix(c('Bitcoin Cash', 'BCH', 'Bitcoin', 'BTC', 'BitConnect', 'BCC', 
           'Dash', 'DASH', 'Ethereum Classic', 'ETC', 'Ethereum', 'ETH', 
           'Iota', 'MIOTA', 'Litecoin', 'LTC', 'Monero', 'XMR', 'Nem', 'XEM', 
           'Neo', 'NEO', 'Numeraire', 'NMR', 'Omisego', 'OMG', 'Qtum', 'QTUM', 
           'Ripple', 'XRP', 'Stratis', 'STRAT', 'Waves', 'WAVES'), 17, 2, byrow=TRUE)
currencies_labels = currencies_labels_tickers[, 1]
currencies_tickers = currencies_labels_tickers[, 2]
num_currencies = dim(currencies_labels_tickers)[1]
currencies_labels_and_tickers = vector("list", num_currencies)
for (i in seq(1,num_currencies)) {
  currency_label = currencies_labels[i]
  currency_ticker = currencies_tickers[i]
  currencies_labels_and_tickers[i] = sprintf("%s (%s)", currency_label, currency_ticker)
}
# Merge the tables into one.
closing_values = NULL
closing_values = merge(bitcoin_cash[c('Date', 'Close')], bitcoin[c('Date', 'Close')],
                       by='Date', all=TRUE, suffixes = c('', '.bitcoin'))
for (cryptocurrency_table_name in cryptocurrency_table_names) {
  if (cryptocurrency_table_name %in% c('bitcoin_cash', 'bitcoin'))
    next
  cryptocurrency_table = lapply(cryptocurrency_table_name, get)[[1]]
  assign('closing_values',
         merge(closing_values, cryptocurrency_table[c('Date', 'Close')], by='Date', all=TRUE, 
               suffixes = c('', sprintf('.%s', cryptocurrency_table_name))))
}
dim(closing_values)

# Change the column names.
colnames(closing_values) = c('Date', currencies_labels_and_tickers)
# head(closing_values, 5)

# If we read from the CSV files, reformat the dates.
if (READ_FROM_CSV) {
  to_datetime = function(date) {
    as.Date(date,"%b %d, %Y")
  }
  reformat_date = function(date) {
    as.POSIXct(date, format="%b %d, %Y")
  }
  closing_values['Date'] = lapply(closing_values['Date'], reformat_date)
}
closing_values = closing_values[order(closing_values['Date']),]
#head(closing_values, 5)

# Value Trends
dir.create(file.path('figures'), showWarnings = FALSE)
for (i in seq(1,num_currencies)) {
  currency_label_ticker = currencies_labels_and_tickers[[i]]
  currency_closing_plotting = melt(closing_values[c("Date", currency_label_ticker)], id.var='Date', variable.name='Currency')
  # Remove NA values.
  currency_closing_plotting = currency_closing_plotting[!(is.na(currency_closing_plotting$value)),]
  # head(currency_closing_plotting)
  currency_label = currencies_labels[i]
  currency_ticker = currencies_tickers[i]
  currency_label_no_spaces = gsub(" ", "_", currency_label)
  currency_value_plot = 
    ggplot(currency_closing_plotting, aes(Date,value,group=1)) + 
    geom_line(aes(colour=Currency)) +
    scale_x_datetime(breaks = pretty(currency_closing_plotting$Date,n=8),labels = date_format("%Y-%m")) +
    ylab("Close") +
    ggtitle(sprintf("%s (%s) Closing Value", 
                    currency_label, currency_ticker)) +
    # Center the title
    theme(plot.title = element_text(hjust = 0.5),
          # Remove legend on right.
          legend.position="none")
  # Save the plot.
  filename = sprintf("%s_price_trend.png", currency_label_no_spaces, showWarnings = FALSE)
  dir.create(file.path('figures', currency_label_no_spaces), showWarnings = FALSE)
  ggsave(filename, path=sprintf('figures/%s', currency_label_no_spaces))
}

# Value Correlations

# Convert POSIXct dates to strings.
std_date_format = "%Y-%m-%d"
date_to_str = function(date) {
  strftime(date, format=std_date_format)
}
# Convert strings to POSIXlt dates.
str_to_datelt = function(date_str) {
  as.POSIXlt(date_str, format=std_date_format)
}

# Make the 'Date' column the index after convering dates to strings.
prices = closing_values
prices['Date'] = lapply(prices['Date'], date_to_str)
prices = prices[,!names(prices) %in% c("Date")]
row.names(prices) = closing_values[,"Date"]

price_correlations = cor(prices, method="pearson", use="pairwise.complete.obs")
price_correlations_plotting = melt(price_correlations)
# Ensure plotting order is the same as column order, which is 
# the same as in the Python version of this figure.
price_correlations_plotting$Var1 =
  factor(price_correlations_plotting$Var1,
         levels=c('Numeraire (NMR)', 'Ripple (XRP)', 'Ethereum Classic (ETC)', 
                  'Stratis (STRAT)', 'BitConnect (BCC)',
                  'Waves (WAVES)', 'Ethereum (ETH)', 'Nem (XEM)', 'Neo (NEO)',
                  'Bitcoin (BTC)', 'Litecoin (LTC)', 'Dash (DASH)', 'Monero (XMR)',
                  'Bitcoin Cash (BCH)', 'Omisego (OMG)', 'Iota (MIOTA)', 'Qtum (QTUM)'))
price_correlations_plotting$Var2 =
  factor(price_correlations_plotting$Var2,
         levels=rev(c('Numeraire (NMR)', 'Ripple (XRP)', 'Ethereum Classic (ETC)', 
                  'Stratis (STRAT)', 'BitConnect (BCC)',
                  'Waves (WAVES)', 'Ethereum (ETH)', 'Nem (XEM)', 'Neo (NEO)',
                  'Bitcoin (BTC)', 'Litecoin (LTC)', 'Dash (DASH)', 'Monero (XMR)',
                  'Bitcoin Cash (BCH)', 'Omisego (OMG)', 'Iota (MIOTA)', 'Qtum (QTUM)')))
ggplot(price_correlations_plotting, aes(Var1, Var2))+ 
  geom_tile(aes(fill=value), colour="white") +
  geom_text(aes(label = round(value, 2)), size=2) +
  theme(# Orient x-axis tick labels vertically.
        axis.text.x = element_text(angle=90, hjust=1)) +
  xlab("Name") +
  ylab("Name") +
  # Remove legend title.
  labs(fill="")
filename = "correlations.png"
ggsave(filename, path='figures')

# Removing Currencies with Short Histories

# See where values are absent and keep only currencies with reasonably lengthy histories.
is_na_prices = as.data.frame(is.na(prices))
is_na_prices['Date'] = row.names(is_na_prices)
is_na_prices = melt(is_na_prices, id.vars='Date', variable.name='Currency')
is_na_prices['value'] = lapply(is_na_prices['value'], as.numeric)
is_na_prices['Date'] = closing_values["Date"]

# Reverse date order to match Python figure.
# Credit to https://stackoverflow.com/a/43626186/5449970.
c_trans = function(a, b, breaks = b$breaks, format = b$format) {
  a = as.trans(a)
  b = as.trans(b)
  
  name = paste(a$name, b$name, sep = "-")
  
  trans = function(x) a$trans(b$trans(x))
  inv = function(x) b$inverse(a$inverse(x))
  
  trans_new(name, trans, inv, breaks, format)
}
rev_date = c_trans("reverse", "time")
ggplot(is_na_prices, aes(Currency, Date, group=1)) + 
  geom_tile(aes(fill=value)) +#, colour="white") +
  theme(# Orient x-axis tick labels vertically.
    axis.text.x = element_text(angle=90, hjust=1, size=6),
    axis.text.y = element_text(size=6)) +
  scale_y_continuous(trans = rev_date) +
  # Remove legend title.
  labs(fill="")
filename = "absent_values.png"
ggsave(filename, path='figures')

currencies_labels_tickers_to_remove = 
  matrix(c('Bitcoin Cash', 'BCH', 'BitConnect', 'BCC', 'Ethereum Classic', 'ETC',
           'Iota', 'MIOTA', 'Neo', 'NEO', 'Numeraire', 'NMR', 'Omisego', 'OMG',
           'Qtum', 'QTUM', 'Stratis', 'STRAT', 'Waves', 'WAVES'), 10, 2, byrow=TRUE)
currencies_labels_to_remove = currencies_labels_tickers_to_remove[, 1]
currencies_tickers_to_remove = currencies_labels_tickers_to_remove[, 2]
num_currencies_to_remove = dim(currencies_labels_tickers_to_remove)[1]
currencies_labels_and_tickers_to_remove = vector("list", num_currencies_to_remove)
for (i in seq(1,num_currencies_to_remove)) {
  currency_label_to_remove = currencies_labels_to_remove[i]
  currency_ticker_to_remove = currencies_tickers_to_remove[i]
  currencies_labels_and_tickers_to_remove[i] = sprintf("%s (%s)", currency_label_to_remove, currency_ticker_to_remove)
}
subset_prices = prices[!(names(prices) %in% currencies_labels_and_tickers_to_remove)]
subset_num_currencies = dim(subset_prices)[2]
subset_currencies_labels = vector("list", subset_num_currencies)
subset_currencies_tickers = vector("list", subset_num_currencies)
subset_currencies_labels_and_tickers = vector("list", subset_num_currencies)
subset_current_currency_index = 1
for (i in seq(1,num_currencies)) {
  currency_label = currencies_labels[i]
  currency_ticker = currencies_tickers[i]
  if (!(currency_label %in% currencies_labels_to_remove)) {
    subset_currencies_labels[subset_current_currency_index] = 
      currency_label
    subset_currencies_tickers[subset_current_currency_index] = 
      currency_ticker
    subset_current_currency_label = subset_currencies_labels[subset_current_currency_index]
    subset_current_currency_ticker = subset_currencies_tickers[subset_current_currency_index]
    subset_currencies_labels_and_tickers[subset_current_currency_index] = 
      sprintf("%s (%s)", subset_current_currency_label, 
              subset_current_currency_ticker)
    subset_current_currency_index = subset_current_currency_index + 1
  }
}
subset_prices_nonan = subset_prices[complete.cases(subset_prices),]

# Volatility Examination

num_non_nan_days = dim(subset_prices_nonan)[1]
# Find the returns.
returns = subset_prices_nonan
for (currency in colnames(subset_prices_nonan)) {
  returns[,currency] = Delt(subset_prices_nonan[,currency])[,1]
}
# Find the standard deviations in returns.
returns_2017 = returns[row.names(returns) >= "2017-01-01" & row.names(returns) < "2018-01-01",]
returns_2017_melted = melt(returns_2017)
returns_std_2017 = ddply(returns_2017_melted, c('variable'), summarize, sd = sd(value))
returns_std_2017 = returns_std_2017[with(returns_std_2017, order(-sd)),]
# Reset row names (indices).
row.names(returns_std_2017) = NULL
# Ensure plotting order is same as column order.
returns_std_2017$variable = factor(returns_std_2017$variable, levels=returns_std_2017$variable)
ggplot(returns_std_2017, aes(variable, fill=variable)) + 
  scale_y_continuous(breaks=seq(0,0.14,0.02)) +
  geom_bar(aes(weight=sd)) +
  xlab("Name") +
  ylab("Volatility") +
  ggtitle("Volatility (2017)") +
  # Center the title
  theme(plot.title = element_text(hjust = 0.5),
      # Make currency label text smaller.
      axis.text.x = element_text(size=7),
      # Remove legend on right.
      legend.position="none")
filename = "volatility.png"
ggsave(filename, path='figures')

# Data Extraction

data = subset_prices_nonan
dates = row.names(data)
row.names(data) = NULL
# head(data)

# We will predict closing prices based on these numbers of days preceding the date of prediction.
# The max `window_size` is `num_non_nan_days`, but some lower values may result in poor models due to small test sets
# during cross validation, and possibly even training failures due to empty test sets for even larger values.
window_sizes = seq(30, 360, 30) # Window sizes measured in days - approximately 1 to 12 months.
for (window_size in window_sizes) {
  print(sprintf("Predicting prices with a window of %s days of preceding currency values", window_size))
  num_windows = dim(data)[1] - window_size
  X = matrix(nrow=num_windows, ncol=subset_num_currencies * window_size, byrow=TRUE)
  for (i in seq(num_windows)) {
    window_vec = as.vector(t(data.matrix(data[seq(i,i+window_size-1),])))
    X[i,] = as.vector(t(data.matrix(data[seq(i,i+window_size-1),])))
  }
  X = as.data.frame(X)
  y = data[seq(window_size+1,dim(data)[1]),]
  # Spaces and parenthesis not tolerated in at least the rfsrc regressor,
  # so use only labels instead of labels and tickers as column names.
  colnames(y) = subset_currencies_labels
  # Merge X and y for at least the rfsrc regressor.
  row.names(y) = NULL
  X_y = merge(x=X,y=y,by="row.names")
  X_y= X_y[,!(names(X_y) %in% c('Row.names'))]
  
  # Model Training
  library(caret)
  set.seed(0)
  train_control = trainControl(method="repeatedcv", number=10, repeats=3)
  
  # Ensemble models
  
  # Extra-Trees regressor
  # Allocate 4 GiB of memory because extra trees is memory intensive.
  options( java.parameters = "-Xmx4g" )
  library(parallel)
  num_cores = detectCores()
  params_extra_trees = expand.grid(mtry = c(ncol(X), sqrt(ncol(X)), log(ncol(X),2)),
                                   numRandomCuts = 1)
  
  # Random forest regressor
  # Takes too long to train in R.
  # params_rf = expand.grid(mtry = c(ncol(X), sqrt(ncol(X)), log(ncol(X),2)))
  
  # Neighbors models
  # k = seq(5,15) would match the Python code, but takes too long to train
  # Also, this model outputs constant values for some reason.
  # params_knn = expand.grid(k = seq(5,5))
  
  model_names = c('extraTrees')#c('extraTrees', 'knn')
                #c('extraTrees', 'rf', 'knn')
  param_sets = list()
  param_sets[[1]] = params_extra_trees
  # param_sets[[2]] = params_knn
  
  # Scores and the corresponding models
  scores = vector(length = length(model_names))
  models_built = vector(length = length(model_names))
  
  for (i in seq(length(model_names))) {
    # CARET train() does not support multi-output models,
    # so we will train one regressor for each currency.
    model = vector(length = dim(y)[2])
    # The average score for the regressors that comprise this model.
    score = 0
    num_cols = dim(y)[2]
    for (j in seq(num_cols)) {
      colname = colnames(y)[j]
      submodel = train(x = X,
                       y = y[[colname]],
                       method = model_names[i],
                       metric = "Rsquared",
                       trControl = train_control,
                       tuneGrid = param_sets[[i]],
                       # ExtraTrees parameters
                       numThreads = num_cores/2)    
      score = score + submodel$results$Rsquared[as.numeric(row.names(submodel$bestTune))]
      model[j] = lapply("submodel", get)
    }
    score = score / num_cols
    scores[i] = score
    models_built[i] = lapply("model", get)
  }
  score_model_tuples = data.frame(score = I(scores), models = I(models_built))
  best_score = score_model_tuples[order(score_model_tuples$score)[1],1][[1]]
  model = score_model_tuples[order(score_model_tuples$score)[1],2][[1]]
  print(sprintf("Best model and score for window size %s: %s %s", window_size, model[[1]]$method, best_score))
  
  # Validation and Visualization
  multioutput_predict = function(submodels, X) {
    out_mat = matrix(nrow=dim(X)[1], ncol=length(submodels), byrow=FALSE)
    # Acquire predictions for the currencies from the regressors.
    for (i in seq(length(submodels))) {
      out_mat[,i] = predict(submodels[i],X)[[1]]
    }
    return(out_mat)
  }
  pred = multioutput_predict(model,X)
  colnames(pred) = colnames(y)
  pred = as.data.frame(pred)
  pred_plotting = pred
  pred_plotting['Date'] = dates[(window_size+1):length(dates)]
  pred_plotting['Date'] = lapply(pred_plotting['Date'],str_to_datelt)
  
  # Plot the actual values along with the predictions. They should overlap.
  subset_num_cols = 2
  subset_num_rows = ceiling(subset_num_currencies / subset_num_cols)
  plots = list(length = subset_num_currencies)
  for (i in seq(subset_num_currencies)) {
    currency_label = subset_currencies_labels[[i]]
    currency_label_no_spaces = gsub(" ", "_", currency_label)
    currency_ticker = subset_currencies_tickers[[i]]
    plots[[i]] = 
      ggplot(data = data.frame()) +
      geom_line(aes(pred_plotting[,'Date'],y[,currency_label], group=1), color="blue", alpha=0.5) +
      geom_line(aes(pred_plotting[,'Date'],pred_plotting[,currency_label], group=1), color="red", alpha=0.5) +
      scale_x_datetime(breaks = pretty(pred_plotting$Date,n=8),labels = date_format("%Y-%m")) +
      xlab("Date") +
      ylab("Close") +
      ggtitle(sprintf("%s (%s) Closing Value (%i day window)",
                      currency_label, currency_ticker, window_size)) +
      # Center the title
      theme(plot.title = element_text(hjust = 0.5))
    currency_figures_subdir = sprintf('%s/%s', figures_dir, currency_label_no_spaces)
    ggsave(filename = sprintf('%s_validation_%i.png',currency_label_no_spaces, window_size),
           path = currency_figures_subdir)
  }
  # All collective plots show the results for Ripple for each cryptocurrency.
  # Ripple is the last plot in the set, but I don't know why this happens.
  collective_plot = 
    do.call(arrangeGrob, 
            c(plots, nrow = subset_num_rows))
  ggsave(filename = sprintf('validation_%i.png', window_size),
         path = figures_dir,
         plot = collective_plot,
         scale = 2)
  
  # Get the model's predictions for the rest of 2017 and 2018.
  last_data_date = closing_values[nrow(closing_values),'Date']
  first_extrapolation_date = last_data_date + 24*60*60
  last_extrapolation_date = str_to_datelt('2018-12-31')
  extrapolation_dates = seq(first_extrapolation_date,
                            last_extrapolation_date,
                            24*60*60)
  num_extrapolation_dates = length(extrapolation_dates)
  extrapolation_X = matrix(0, nrow = num_extrapolation_dates, ncol = subset_num_currencies * window_size, byrow=TRUE)
  colnames(extrapolation_X) = colnames(X)
  extrapolation_y = matrix(0, nrow = num_extrapolation_dates, ncol = subset_num_currencies, byrow=TRUE)
  
  # First `window_size` windows contain known values.
  given_prices = subset_prices_nonan
  row.names(given_prices) = NULL
  given_prices = as.vector(t(given_prices[(nrow(given_prices)-window_size+1):nrow(given_prices),]))
  extrapolation_X[1,] = given_prices
  extrapolation_y[1,] = multioutput_predict(model, extrapolation_X)[1,]
  for (i in seq(window_size-1)) {
    given_prices = as.vector(t(subset_prices_nonan[(nrow(subset_prices_nonan)-window_size+i+1):nrow(subset_prices_nonan),]))
    previous_predicted_prices = as.vector(t(extrapolation_y[1:i,]))
    extrapolation_X[i+1,] = append(given_prices, previous_predicted_prices)
    extrapolation_y[i+1,] = multioutput_predict(model, extrapolation_X)[i+1,]
  }
  # Remaining windows contain only predicted values (predicting based on previous predictions).
  for (i in seq(window_size, num_extrapolation_dates-1)) {
    previous_predicted_prices = as.vector(t(extrapolation_y[(i-window_size+1):i,]))
    extrapolation_X[i+1,] = previous_predicted_prices
    extrapolation_y[i+1,] = multioutput_predict(model, extrapolation_X)[i+1,]
  }
  
  # Plot the predictions for the rest of 2017 and 2018.
  for (i in seq(subset_num_currencies)) {
    currency_label = subset_currencies_labels[[i]]
    currency_label_no_spaces = gsub(" ", "_", currency_label)
    currency_ticker = subset_currencies_tickers[[i]]
    plots[[i]] = 
      ggplot(data = data.frame()) +
      geom_line(aes(extrapolation_dates,extrapolation_y[,i], group=1), color="red") +
      scale_x_datetime(breaks = pretty(extrapolation_dates,n=8),labels = date_format("%Y-%m")) +
      xlab("Date") +
      ylab("Close") +
      ggtitle(sprintf("%s (%s) Closing Value (%i day window)",
                      currency_label, currency_ticker, window_size)) +
      # Center the title
      theme(plot.title = element_text(hjust = 0.5))
    currency_figures_subdir = sprintf('%s/%s', figures_dir, currency_label_no_spaces)
    ggsave(filename = sprintf('%s_predictions_%i.png',currency_label_no_spaces, window_size),
           path = currency_figures_subdir)
  }
  collective_plot = 
    do.call(arrangeGrob, 
            c(plots, nrow = subset_num_rows))
  ggsave(filename = sprintf('predictions_%i.png', window_size),
         path = figures_dir,
         plot = collective_plot,
         scale = 2)
  
  # Plot the predicitons for the rest of 2017 and 2018 along with the actual values for the date range used.
  for (i in seq(subset_num_currencies)) {
    currency_label = subset_currencies_labels[[i]]
    currency_label_no_spaces = gsub(" ", "_", currency_label)
    currency_ticker = subset_currencies_tickers[[i]]
    plots[[i]] = 
      ggplot(data = data.frame()) +
      geom_line(aes(pred_plotting[,'Date'],y[,i], group=1), color="blue") +
      geom_line(aes(extrapolation_dates,extrapolation_y[,i], group=1), color="red") +
      scale_x_datetime(breaks = pretty(extrapolation_dates,n=8),labels = date_format("%Y-%m")) +
      xlab("Date") +
      ylab("Close") +
      ggtitle(sprintf("%s (%s) True + Predicted Closing Value (%i day window)",
                      currency_label, currency_ticker, window_size)) +
      # Center the title
      theme(plot.title = element_text(hjust = 0.5))
    currency_figures_subdir = sprintf('%s/%s', figures_dir, currency_label_no_spaces)
    ggsave(filename = sprintf('%s_actual_plus_predictions_%i.png', currency_label_no_spaces, window_size),
           path = currency_figures_subdir)
  }
  collective_plot = 
    do.call(arrangeGrob, 
            c(plots, nrow = subset_num_rows))
  ggsave(filename = sprintf('actual_plus_predictions_%i.png', window_size),
         path = figures_dir,
         plot = collective_plot,
         scale = 2)
}

library(RMySQL)
library(ggplot2)
library(reshape2)
library(scales)
library(quantmod)
library(plyr)

READ_FROM_CSV = TRUE

bitcoin_cash <- NULL; bitcoin <- NULL; bitconnect <- NULL;
dash <- NULL; ethereum_classic <- NULL; ethereum <- NULL;
iota <- NULL; litecoin <- NULL; monero <- NULL;
nem <- NULL; neo <- NULL; numeraire <- NULL; 
omisego <- NULL; qtum <- NULL; ripple <- NULL;
stratis <- NULL; waves <- NULL;
if (!READ_FROM_CSV) {
  # Data Importing (SQL)
  con <- dbConnect(MySQL(), user='john', 
                   password='Iwbicvi1994mysql', host='localhost',
                   dbname='cryptocurrencies')
  bitcoin_cash <- dbReadTable(conn=con, name='bitcoin_cash')  
  # TODO: Finish these SQL reads.
} else {
  # Data Importing (CSV)
  bitcoin_cash <- read.csv('../data/bitcoin_cash_price.csv')
  # head(bitcoin_cash, 1)
  bitcoin <- read.csv('../data/bitcoin_price.csv')
  # head(bitcoin, 1)
  bitconnect <- read.csv('../data/bitconnect_price.csv')
  # head(bitconnect, 1)
  dash <- read.csv('../data/dash_price.csv')
  # head(dash, 1)
  ethereum_classic <- read.csv('../data/ethereum_classic_price.csv')
  # head(ethereum_classic, 1)
  ethereum <- read.csv('../data/ethereum_price.csv') 
  # head(ethereum, 1)
  iota <- read.csv('../data/iota_price.csv')
  # head(iota, 1)
  litecoin <- read.csv('../data/litecoin_price.csv')
  # head(litecoin, 1)
  monero <- read.csv('../data/monero_price.csv')
  # head(monero, 1)
  nem <- read.csv('../data/nem_price.csv')
  # head(nem, 1)
  neo <- read.csv('../data/neo_price.csv')
  # head(neo, 1)
  numeraire <- read.csv('../data/numeraire_price.csv')
  # head(numeraire, 1)
  omisego <- read.csv('../data/omisego_price.csv')
  # head(omisego, 1)
  qtum <- read.csv('../data/qtum_price.csv')
  # head(qtum, 1)
  ripple <- read.csv('../data/ripple_price.csv')
  # head(ripple, 1)
  stratis <- read.csv('../data/stratis_price.csv')
  # head(stratis, 1)
  waves <- read.csv('../data/waves_price.csv')
  # head(waves, 1) 
}
currencies_labels_tickers = 
  matrix(c('Bitcoin Cash', 'BCH', 'Bitcoin', 'BTC', 'BitConnect', 'BCC', 
           'Dash', 'DASH', 'Ethereum Classic', 'ETC', 'Ethereum', 'ETH', 
           'Iota', 'MIOTA', 'Litecoin', 'LTC', 'Monero', 'XMR', 'Nem', 'XEM', 
           'Neo', 'NEO', 'Numeraire', 'NMR', 'Omisego', 'OMG', 'Qtum', 'QTUM', 
           'Ripple', 'XRP', 'Stratis', 'STRAT', 'Waves', 'WAVES'), 17, 2, byrow=TRUE)
currencies_labels = currencies_labels_tickers[, 1]
currencies_tickers = currencies_labels_tickers[, 2]
num_currencies = dim(currencies_labels_tickers)[1]
currencies_labels_and_tickers <- vector("list", num_currencies)
for (i in seq(1,num_currencies)) {
  currency_label <- currencies_labels[i]
  currency_ticker <- currencies_tickers[i]
  currencies_labels_and_tickers[i] <- sprintf("%s (%s)", currency_label, currency_ticker)
}
# TODO: Do this in a loop.
closing_values <- NULL
closing_values <- merge(bitcoin_cash[c('Date', 'Close')], bitcoin[c('Date', 'Close')],
                    by='Date', all=TRUE, suffixes = c('', '.bitcoin'))
closing_values <- merge(closing_values, bitconnect[c('Date', 'Close')], by='Date', all=TRUE, 
                    suffixes = c('', '.bitconnect'))
closing_values <- merge(closing_values, dash[c('Date', 'Close')], by='Date', all=TRUE, 
                    suffixes = c('', '.dash'))
closing_values <- merge(closing_values, ethereum_classic[c('Date', 'Close')], by='Date', all=TRUE, 
                    suffixes = c('', '.ethereum_classic'))
closing_values <- merge(closing_values, ethereum[c('Date', 'Close')], by='Date', all=TRUE, 
                    suffixes = c('', '.ethereum'))
closing_values <- merge(closing_values, iota[c('Date', 'Close')], by='Date', all=TRUE, 
                    suffixes = c('', '.iota'))
closing_values <- merge(closing_values, litecoin[c('Date', 'Close')], by='Date', all=TRUE, 
                    suffixes = c('', '.litecoin'))
closing_values <- merge(closing_values, monero[c('Date', 'Close')], by='Date', all=TRUE, 
                    suffixes = c('', '.monero'))
closing_values <- merge(closing_values, nem[c('Date', 'Close')], by='Date', all=TRUE, 
                    suffixes = c('', '.nem'))
closing_values <- merge(closing_values, neo[c('Date', 'Close')], by='Date', all=TRUE, 
                    suffixes = c('', '.neo'))
closing_values <- merge(closing_values, numeraire[c('Date', 'Close')], by='Date', all=TRUE, 
                    suffixes = c('', '.numeraire'))
closing_values <- merge(closing_values, omisego[c('Date', 'Close')], by='Date', all=TRUE, 
                    suffixes = c('', '.omisego'))
closing_values <- merge(closing_values, qtum[c('Date', 'Close')], by='Date', all=TRUE, 
                    suffixes = c('', '.qtum'))
closing_values <- merge(closing_values, ripple[c('Date', 'Close')], by='Date', all=TRUE, 
                    suffixes = c('', '.ripple'))
closing_values <- merge(closing_values, stratis[c('Date', 'Close')], by='Date', all=TRUE, 
                    suffixes = c('', '.stratis'))
closing_values <- merge(closing_values, waves[c('Date', 'Close')], by='Date', all=TRUE, 
                    suffixes = c('', '.waves'))

# Change the column names.
colnames(closing_values) <- c('Date', currencies_labels_and_tickers)
#head(closing_values, 1)

# If we read from the CSV files, reformat the dates.
if (READ_FROM_CSV) {
  to_datetime <- function(date) {
    as.Date(date,"%b %d, %Y")
  }
  reformat_date <- function(date) {
    as.POSIXct(date, format="%b %d, %Y")
  }
  closing_values['Date'] <- lapply(closing_values['Date'], reformat_date)
}
closing_values <- closing_values[order(closing_values['Date']),]
#head(closing_values, 5)

#prices <- data.frame(bitcoin[,'Close'], bitcoin_cash[,'Close'], bitconnect[,'Close'], 
#                     dash[,'Close'], ethereum[,'Close'], ethereum_classic[,'Close'], 
#                     iota[,'Close'], litecoin[,'Close'], monero[,'Close'], nem[,'Close'], 
#                     neo[,'Close'], numeraire[,'Close'], omisego[,'Close'], qtum[,'Close'], 
#                     ripple[,'Close'], stratis[,'Close'], waves[,'Close'])#, 
#                     #row.names=currencies_labels_and_tickers)
#names(prices) <- currencies_labels_and_tickers
#prices[1]

# Value Trends
dir.create(file.path('figures'), showWarnings = FALSE)
for (i in seq(1,num_currencies)) {
  currency_label_ticker <- currencies_labels_and_tickers[[i]]
  currency_closing_plotting <- melt(closing_values[c("Date", currency_label_ticker)], id.var='Date', variable.name='Currency')
  # Remove NA values.
  currency_closing_plotting <- currency_closing_plotting[!(is.na(currency_closing_plotting$value)),]
  # head(currency_closing_plotting)
  currency_label <- currencies_labels[i]
  currency_ticker <- currencies_tickers[i]
  currency_label_no_spaces <- gsub(" ", "_", currency_label)
  currency_value_plot <- 
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
  filename <- sprintf("%s_price_trend.png", currency_label_no_spaces, showWarnings = FALSE)
  dir.create(file.path('figures', currency_label_no_spaces), showWarnings = FALSE)
  # ggsave(filename, path=sprintf('figures/%s', currency_label_no_spaces))
}

# Value Correlations

# Convert POSIXct dates to strings.
std_date_format <- "%Y-%m-%d"
date_to_str <- function(date) {
  strftime(date, format=std_date_format)
}
# Convert strings to POSIXlt dates.
str_to_datelt <- function(date_str) {
  as.POSIXlt(date_str, format=std_date_format)
}

# Make the 'Date' column the index after convering dates to strings.
prices <- closing_values
prices['Date'] <- lapply(prices['Date'], date_to_str)
prices <- prices[,!names(prices) %in% c("Date")]
row.names(prices) <- closing_values[,"Date"]

price_correlations <- cor(prices, method="pearson", use="pairwise.complete.obs")
price_correlations_plotting <- melt(price_correlations)
# Ensure plotting order is the same as column order, which is 
# the same as in the Python version of this figure.
price_correlations_plotting$Var1 <-
  factor(price_correlations_plotting$Var1,
         levels=c('Numeraire (NMR)', 'Ripple (XRP)', 'Ethereum Classic (ETC)', 
                  'Stratis (STRAT)', 'BitConnect (BCC)',
                  'Waves (WAVES)', 'Ethereum (ETH)', 'Nem (XEM)', 'Neo (NEO)',
                  'Bitcoin (BTC)', 'Litecoin (LTC)', 'Dash (DASH)', 'Monero (XMR)',
                  'Bitcoin Cash (BCH)', 'Omisego (OMG)', 'Iota (MIOTA)', 'Qtum (QTUM)'))
price_correlations_plotting$Var2 <-
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
filename <- "correlations.png"
# ggsave(filename, path='figures')

# Removing Currencies with Short Histories

# See where values are absent and keep only currencies with reasonably lengthy histories.
is_na_prices <- as.data.frame(is.na(prices))
is_na_prices['Date'] <- row.names(is_na_prices)
is_na_prices <- melt(is_na_prices, id.vars='Date', variable.name='Currency')
is_na_prices['value'] <- lapply(is_na_prices['value'], as.numeric)
is_na_prices['Date'] <- closing_values["Date"]

# Reverse the order of records for plotting
# rownames(is_na_prices_reversed) <- NULL
# Reverse date order to match Python figure.
# Credit to https://stackoverflow.com/a/43626186/5449970.
c_trans <- function(a, b, breaks = b$breaks, format = b$format) {
  a <- as.trans(a)
  b <- as.trans(b)
  
  name <- paste(a$name, b$name, sep = "-")
  
  trans <- function(x) a$trans(b$trans(x))
  inv <- function(x) b$inverse(a$inverse(x))
  
  trans_new(name, trans, inv, breaks, format)
}
rev_date <- c_trans("reverse", "time")
ggplot(is_na_prices, aes(Currency, Date, group=1)) + 
  geom_tile(aes(fill=value)) +#, colour="white") +
  theme(# Orient x-axis tick labels vertically.
    axis.text.x = element_text(angle=90, hjust=1, size=6),
    axis.text.y = element_text(size=6)) +
  scale_y_continuous(trans = rev_date) +
  # Remove legend title.
  labs(fill="")
filename <- "absent_values.png"
# ggsave(filename, path='figures')

currencies_labels_tickers_to_remove <- 
  matrix(c('Bitcoin Cash', 'BCH', 'BitConnect', 'BCC', 'Ethereum Classic', 'ETC',
           'Iota', 'MIOTA', 'Neo', 'NEO', 'Numeraire', 'NMR', 'Omisego', 'OMG',
           'Qtum', 'QTUM', 'Stratis', 'STRAT', 'Waves', 'WAVES'), 10, 2, byrow=TRUE)
currencies_labels_to_remove = currencies_labels_tickers_to_remove[, 1]
currencies_tickers_to_remove = currencies_labels_tickers_to_remove[, 2]
num_currencies_to_remove <- dim(currencies_labels_tickers_to_remove)[1]
currencies_labels_and_tickers_to_remove <- vector("list", num_currencies_to_remove)
for (i in seq(1,num_currencies_to_remove)) {
  currency_label_to_remove <- currencies_labels_to_remove[i]
  currency_ticker_to_remove <- currencies_tickers_to_remove[i]
  currencies_labels_and_tickers_to_remove[i] <- sprintf("%s (%s)", currency_label_to_remove, currency_ticker_to_remove)
}
subset_prices <- prices[!(names(prices) %in% currencies_labels_and_tickers_to_remove)]
subset_num_currencies <- dim(subset_prices)[2]
subset_currencies_labels <- vector("list", subset_num_currencies)
subset_currencies_tickers <- vector("list", subset_num_currencies)
subset_currencies_labels_and_tickers <- vector("list", subset_num_currencies)
subset_current_currency_index <- 1
for (i in seq(1,num_currencies)) {
  currency_label <- currencies_labels[i]
  currency_ticker <- currencies_tickers[i]
  if (!(currency_label %in% currencies_labels_to_remove)) {
    subset_currencies_labels[subset_current_currency_index] <- 
      currency_label
    subset_currencies_tickers[subset_current_currency_index] <- 
      currency_ticker
    subset_current_currency_label <- subset_currencies_labels[subset_current_currency_index]
    subset_current_currency_ticker <- subset_currencies_tickers[subset_current_currency_index]
    subset_currencies_labels_and_tickers[subset_current_currency_index] <- 
      sprintf("%s (%s)", subset_current_currency_label, 
              subset_current_currency_ticker)
    subset_current_currency_index <- subset_current_currency_index + 1
  }
}
subset_prices_nonan <- subset_prices[complete.cases(subset_prices),]

# Volatility Examination

num_non_nan_days = dim(subset_prices_nonan)[1]
# Find the returns.
returns <- subset_prices_nonan
for (currency in colnames(subset_prices_nonan)) {
  returns[,currency] <- Delt(subset_prices_nonan[,currency])[,1]
}
# Find the standard deviations in returns.
returns_2017 <- returns[row.names(returns) >= "2017-01-01" & row.names(returns) < "2018-01-01",]
returns_2017_melted <- melt(returns_2017)
returns_std_2017 <- ddply(returns_2017_melted, c('variable'), summarize, sd = sd(value))
returns_std_2017 <- returns_std_2017[with(returns_std_2017, order(-sd)),]
# Reset row names (indices).
row.names(returns_std_2017) <- NULL
# Ensure plotting order is same as column order.
returns_std_2017$variable <- factor(returns_std_2017$variable, levels=returns_std_2017$variable)
ggplot(returns_std_2017, aes(variable, fill=variable)) + 
  scale_y_continuous(breaks=seq(0,0.14,0.02)) +
  geom_bar(aes(weight=sd)) +
  xlab("Name") +
  ylab("Volatility") +
  ggtitle("Volatility (2017)") +
  # Center the title
  theme(plot.title = element_text(hjust = 0.5),
      # Remove legend on right.
      legend.position="none")
filename <- "volatility.png"
# ggsave(filename, path='figures')

# Data Extraction

data <- subset_prices_nonan
dates <- row.names(data)
row.names(data) <- NULL
head(data)

# We will predict closing prices based on these numbers of days preceding the date of prediction.
# The max `window_size` is `num_non_nan_days`, but some lower values may result in poor models due to small test sets
# during cross validation, and possibly even training failures due to empty test sets for even larger values.
window_sizes <- seq(30, 360, 30) # Window sizes measured in days - approximately 1 to 12 months.

window_size <- window_sizes[1]
print(sprintf("Predicting prices with a window of %s days of preceding currency values", window_size))
num_windows <- dim(data)[1] - window_size
X <- matrix(nrow=num_windows, ncol=subset_num_currencies * window_size, byrow=TRUE)
for (i in seq(num_windows)) {
  window_vec <- as.vector(t(data.matrix(data[seq(i,i+window_size-1),])))
  X[i,] <- as.vector(t(data.matrix(data[seq(i,i+window_size-1),])))
}
X <- as.data.frame(X)
y <- data[seq(window_size+1,dim(data)[1]),]
# Spaces and parenthesis not tolerated in at least the rfsrc regressor,
# so use only labels instead of labels and tickers as column names.
colnames(y) <- subset_currencies_labels
# Merge X and y for at least the rfsrc regressor.
row.names(y) <- NULL
X_y = merge(x=X,y=y,by="row.names")
X_y<- X_y[,!(names(X_y) %in% c('Row.names'))]

# Model Training
library(caret)
set.seed(0)
train_control = trainControl(method="repeatedcv", number=10, repeats=3)

# Ensemble models

# TODO: Extra-Trees regressor
# library(extraTrees)
# y_vec = ...
# model_extra_trees = extraTrees(X,y_vec)
# extraTrees()
# Allocate 4 GiB of memory because extra trees is memory intensive.
options( java.parameters = "-Xmx4g" )
library(parallel)
num_cores = detectCores()
params_extra_trees = expand.grid(#ntree = c(500),
                                 #nodesize = c(2,5,10),
                                 mtry = c(ncol(X), sqrt(ncol(X)), log(ncol(X),2)),
                                 numRandomCuts = 1)

# Random forest regressor
# library(randomForest)
# library(randomForestSRC)
# TODO: Try to have same parameters as in Python code.
# model_random_forest = rfsrc(formula = 
#                               Multivar(Bitcoin, Dash, Ethereum, Litecoin, 
#                                        Monero, Nem, Ripple) ~ .,
#                             data = X_y,
#                             ntree = 500,
#                             nodesize = 1)
#nodesize = [1, 2.5, 5]

# Neighbors models
params_knn = expand.grid(k = seq(5,5))
# model_knn = train(form = Bitcoin + Dash + Ethereum + Litecoin + 
#                            Monero + Nem + Ripple ~ .,
#                   data = X_y, 
#                   method = "knn",
#                   metric = "Rsquared",
#                   trControl = train_control,
#                   tuneGrid = params_knn)

models = c('extraTrees', 'knn')#('extraTrees', 'rf', 'knn')
param_sets = list()
param_sets[[1]] = params_extra_trees
param_sets[[2]] = params_knn
# param_sets = c(params_knn)#(params_extra_trees, params_random_forest, params_knn)

# Scores and the corresponding models
scores = vector(length = length(models))
models_built = vector(length = length(models))
# score_model_tuples = data.frame(score=rep(NA,length(models)),
#                                 model=rep(NA,length(models)))

#(nrow=length(models), ncol=2)

for (i in seq(length(models))) {
  print(models[i])
  print(param_sets[[i]])  
  model = train(form = Bitcoin + Dash + Ethereum + Litecoin +
                  Monero + Nem + Ripple ~ .,
                data = X_y,
                method = models[i],
                metric = "Rsquared",
                trControl = train_control,
                tuneGrid = param_sets[[i]],
                # ExtraTrees parameters
                numThreads = num_cores/2)
  score = model$results$Rsquared[as.numeric(row.names(model$bestTune))]
  scores[i] = score
  models_built[i] = lapply("model", get)
  print("scores:"); print(scores)
  print("models_built:"); print(models_built)
}
score_model_tuples = data.frame(score = I(scores), models = I(models_built))
best_score = score_model_tuples[order(score_model_tuples$score)[1],1][[1]]
model = score_model_tuples[order(score_model_tuples$score)[1],2][[1]]
print(sprintf("Best model and score for window size %s: %s %s", window_size, model$method, best_score))

# Validation and Visualization
# Why does the model only output one value for each input vector?
predict(model,X)

for (window_size in window_sizes) {
  
}

library(RMySQL)
library(ggplot2)
library(scales)

READ_FROM_CSV = TRUE

bitcoin_cash <- NULL; bitcoin <- NULL; bitconnect <- NULL;
dash <- NULL; ethereum_classic <- NULL; ethereum <- NULL;
iota <- NULL; litecoin <- NULL; monero <- NULL
nem <- NULL; neo <- NULL; numeraire <- NULL; 
omisego <- NULL; qtum <- NULL; ripple <- NULL
stratis <- NULL; waves <- NULL
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
# Make the 'Date' column the index.
# bitcoin_cash <- data.frame(bitcoin_cash[,-1], row.names = bitcoin_cash[,1])
# bitcoin <- data.frame(bitcoin[,-1], row.names = bitcoin[,1])
# bitconnect <- data.frame(bitconnect[,-1], row.names = bitconnect[,1])
# dash <- data.frame(dash[,-1], row.names = dash[,1])
# ethereum_classic <- data.frame(ethereum_classic[,-1], row.names = ethereum_classic[,1])
# ethereum <- data.frame(ethereum[,-1], row.names = ethereum[,1])
# iota <- data.frame(iota[,-1], row.names = iota[,1])
# litecoin <- data.frame(litecoin[,-1], row.names = litecoin[,1])
# monero <- data.frame(monero[,-1], row.names = monero[,1])
# nem <- data.frame(nem[,-1], row.names = nem[,1])
# neo <- data.frame(neo[,-1], row.names = neo[,1])
# numeraire <- data.frame(numeraire[,-1], row.names = numeraire[,1])
# omisego <- data.frame(omisego[,-1], row.names = omisego[,1])
# qtum <- data.frame(qtum[,-1], row.names = qtum[,1])
# ripple <- data.frame(ripple[,-1], row.names = ripple[,1])
# stratis <- data.frame(stratis[,-1], row.names = stratis[,1])
# waves <- data.frame(waves[,-1], row.names = waves[,1])

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
currencies_labels_and_tickers
#bitcoin[c('Date', 'Close')]
#bitcoin_cash[1,]
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
head(closing_values, 1)

# If we read from the CSV files, reformat the dates.
if (READ_FROM_CSV) {
  to_datetime <- function(date) {
    as.Date(date,"%b %d, %Y")
  }
  reformat_date <- function(date) {
    # as.character(format(as.Date(date,"%b %d, %Y")))
    as.POSIXct(date, format="%b %d, %Y")
  }
  # closing_values['Date'] <- lapply(closing_values['Date'], to_datetime)
  closing_values['Date'] <- lapply(closing_values['Date'], reformat_date)
}
closing_values <- closing_values[order(closing_values['Date']),]

# Make the 'Date' column the index.
#closing_values[!(names(closing_values) %in% c("Date"))][1:5,]
#closing_values <- data.frame(closing_values[,!(names(closing_values) %in% c("Date"))], row.names = closing_values[,"Date"])
#colnames(closing_values) <- currencies_labels_and_tickers
head(closing_values, 5)

#prices <- data.frame(bitcoin[,'Close'], bitcoin_cash[,'Close'], bitconnect[,'Close'], 
#                     dash[,'Close'], ethereum[,'Close'], ethereum_classic[,'Close'], 
#                     iota[,'Close'], litecoin[,'Close'], monero[,'Close'], nem[,'Close'], 
#                     neo[,'Close'], numeraire[,'Close'], omisego[,'Close'], qtum[,'Close'], 
#                     ripple[,'Close'], stratis[,'Close'], waves[,'Close'])#, 
#                     #row.names=currencies_labels_and_tickers)
#names(prices) <- currencies_labels_and_tickers
#prices[1]

# Value Trends
ChickWeight[1:5,]

dat <- matrix(runif(40,1,20),ncol=4) # make data
dat
matplot(dat, type = c("b"),pch=1,col = 1:4) #plot
legend("topleft", legend = 1:4, col=1:4, pch=1) # optional legend

library(reshape2)
# df <- data.frame(time = 1:10,
#                  a = cumsum(rnorm(10)),
#                  b = cumsum(rnorm(10)),
#                  c = cumsum(rnorm(10)))
# head(df)
# df <- melt(df, id.vars = 'time', variable.name = 'series')
# df

dir.create(file.path('figures'), showWarnings = FALSE)
for (i in seq(1,num_currencies)) {
  currency_label_ticker <- currencies_labels_and_tickers[[i]]
  currency_closing_plotting <- melt(closing_values[c("Date", currency_label_ticker)], id.var='Date', variable.name='Currency')
  # Remove NA values.
  currency_closing_plotting <- currency_closing_plotting[!(is.na(currency_closing_plotting$value)),]
  head(currency_closing_plotting)
  currency_value_plot <- 
    ggplot(currency_closing_plotting, aes(Date,value,group=1)) + 
    theme(text = element_text(size=8),
          # Orient x-axis tick labels vertically.
          axis.text.x = element_text(angle=90, hjust=1)) +
    geom_line(aes(colour=Currency)) +
    scale_x_datetime(breaks = date_breaks("4 weeks"), labels = date_format("%Y-%m")) +#"", format="%Y-%b") +
    ylab("")
  # Save the plot.
  currency_label <- currencies_labels[i]
  currency_label_no_spaces <- gsub(" ", "_", currency_label)
  filename <- sprintf("%s_price_trend.png", currency_label_no_spaces, showWarnings = FALSE)
  dir.create(file.path('figures', currency_label_no_spaces), showWarnings = FALSE)
  ggsave(filename, path=sprintf('figures/%s', currency_label_no_spaces))
}

# bitcoin_closing_plotting <- melt(closing_values[c("Date", "Bitcoin (BTC)")], id.var='Date', variable.name='Currency')
# head(bitcoin_closing_plotting)
# ggplot(bitcoin_closing_plotting, aes(Date,value)) + geom_line(aes(colour=Currency))
#closing_values_plotting <- melt(closing_values, id.vars='Date', variable.name='Currency')
#ggplot(closing_values_plotting, aes(Date,value)) + geom_line(aes(colour=Currency))

# Value Correlations

# Convert POSIXct dates to strings.
date_to_str <- function(date) {
  strftime(date, format="%Y-%m-%d")
}
# closing_values['Date'] <- lapply(closing_values['Date'], date_to_str)
# head(closing_values['Date'])

# Make the 'Date' column the index.
prices <- closing_values
prices['Date'] <- lapply(prices['Date'], date_to_str)
prices <- prices[,!names(prices) %in% c("Date")]
row.names(prices) <- closing_values[,"Date"]
head(prices, 1)

price_correlations <- cor(prices, method="pearson", use="pairwise.complete.obs")
price_correlations
# heatmap(price_correlations, symm=TRUE)
price_correlations_plotting <- melt(price_correlations)
price_correlations_plotting
#price_correlations_plotting["pos"] <- cumsum()
ggplot(price_correlations_plotting, aes(Var1, Var2))+ 
  geom_tile(aes(fill=value), colour="white") +
  geom_text(aes(label = round(value, 2)), size=2) +
  theme(# Orient x-axis tick labels vertically.
        axis.text.x = element_text(angle=90, hjust=1)) +
  xlab("Name") +
  ylab("Name")
filename <- "correlations.png"
ggsave(filename, path='figures')

# Removing Currencies with Short Histories

# See where values are absent and keep only currencies with reasonably lengthy histories.
is_na_prices <- as.data.frame(is.na(prices))
is_na_prices['Date'] <- row.names(is_na_prices)
is_na_prices <- melt(is_na_prices, id.vars='Date', variable.name='Currency')
is_na_prices['value'] <- lapply(is_na_prices['value'], as.numeric)
closing_values[,"Date"]
is_na_prices['Date'] <- closing_values["Date"]
class(is_na_prices[1,'Date'])
head(is_na_prices)
ggplot(is_na_prices, aes(Currency, Date, group=1)) + 
  geom_tile(aes(fill=value)) +#, colour="white") +
  theme(# Orient x-axis tick labels vertically.
    axis.text.x = element_text(angle=90, hjust=1, size=6),
    axis.text.y = element_text(size=6)) +
  scale_y_datetime(breaks = date_breaks("9 weeks"), labels = date_format("%Y-%m"))
filename <- "absent_values.png"
ggsave(filename, path='figures')

# currencies_labels_tickers_to_remove <- c(c('Bitcoin Cash', 'BCH'), ['BitConnect', 'BCC'], ['Ethereum Classic', 'ETC'],
#                                          ['Iota', 'MIOTA'], ['Neo', 'NEO'], ['Numeraire', 'NMR'], ['Omisego', 'OMG'],
#                                          ['Qtum', 'QTUM'], ['Stratis', 'STRAT'], ['Waves', 'WAVES'])
# currencies_labels_to_remove = currencies_labels_tickers_to_remove[:, 0]
# currencies_tickers_to_remove = currencies_labels_tickers_to_remove[:, 1]
# currencies_labels_and_tickers_to_remove = ["{} ({})".format(currencies_label, currencies_ticker)
#                                            for currencies_label, currencies_ticker in
#                                            zip(currencies_labels_to_remove, currencies_tickers_to_remove)]

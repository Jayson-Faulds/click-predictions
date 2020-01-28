library(data.table)
library(dplyr)
library(randomForest)

# import the data, add column names
df <- fread('ProjectTrainingData-1MM.csv')
names <- c('id', 'click', 'hour', 'C1', 'banner_pos', 'site_id', 'site_domain', 'site_category', 'app_id', 'app_domain',
           'app_category', 'device_id', 'device_ip', 'device_model', 'device_type', 'device_conn_type', 'C14', 'C15', 
           'C16', 'C17', 'C18', 'C19', 'C20', 'C21')
colnames(df) <- names

# Extracts the hour from the hour column
df$hour <- as.character(df$hour)
df$hour <- substr(df$hour, 7, 8)

# get rid of ID column
df <- df[, -1]

# We need to convert the non-numeric categories into numbers to run our algorithms
# Using one hot encoding on most of these categories would create way too many dummies
# For columns with more than 10 distinct categories, we will try to do some feature engineering
# to limit the number of dummies we create

# Gets the average ad clicks for each unique site id
df <- df %>%
  group_by(site_id) %>%
  mutate(avg_clicks = sum(click)/n())

# Here, if the site has an average click in the quantile ranges below, they get put into groups 1-5
# Anything below the median is classified in group 0 (Around half the websites get zero clicks)
unique_sites <- distinct_at(df, .vars = 'site_id', .keep_all = T)
unique_sites$new_site_id <- 0
q5 <- quantile(unique_sites$avg_clicks, 0.5)
q6 <- quantile(unique_sites$avg_clicks, 0.6)
q7 <- quantile(unique_sites$avg_clicks, 0.7)
q8 <- quantile(unique_sites$avg_clicks, 0.8)
q9 <- quantile(unique_sites$avg_clicks, 0.9)
unique_sites$new_site_id[unique_sites$avg_clicks < q6 & unique_sites$avg_clicks > q5] <- 1
unique_sites$new_site_id[unique_sites$avg_clicks < q7 & unique_sites$avg_clicks > q6] <- 2
unique_sites$new_site_id[unique_sites$avg_clicks < q8 & unique_sites$avg_clicks > q7] <- 3
unique_sites$new_site_id[unique_sites$avg_clicks < q9 & unique_sites$avg_clicks > q8] <- 4
unique_sites$new_site_id[unique_sites$avg_clicks > q9] <- 5

df <- left_join(df, unique_sites[, c(5, 25)], by = 'site_id')
rm(unique_sites)

# We realized that never-before-seen websites in the test data would cause errors with prediction
# To account for this, we identified websites with low frequency in the training as an "other" category
# Now, new websites in the test data will get coded as "low_freq" rather than a 0, which would probably cause bias
df <- df %>%
  group_by(site_id) %>%
  mutate(count = n())

# Here we define low frequency websites as getting visited on average twice a day (20/10==2)
df$new_site_id[df$count <= 20] <- 'low_freq'

# We will have to repeat this same process for much of the columns in df
df <- df %>%
  group_by(site_domain) %>%
  mutate(avg_clicks = sum(click)/n())

unique_site_domains <- distinct_at(df, .vars = 'site_domain', .keep_all = T)
unique_site_domains$new_site_domain <- 0
q5 <- quantile(unique_site_domains$avg_clicks, 0.5)
q6 <- quantile(unique_site_domains$avg_clicks, 0.6)
q7 <- quantile(unique_site_domains$avg_clicks, 0.7)
q8 <- quantile(unique_site_domains$avg_clicks, 0.8)
q9 <- quantile(unique_site_domains$avg_clicks, 0.9)
unique_site_domains$new_site_domain[unique_site_domains$avg_clicks < q6 & unique_site_domains$avg_clicks > q5] <- 1
unique_site_domains$new_site_domain[unique_site_domains$avg_clicks < q7 & unique_site_domains$avg_clicks > q6] <- 2
unique_site_domains$new_site_domain[unique_site_domains$avg_clicks < q8 & unique_site_domains$avg_clicks > q7] <- 3
unique_site_domains$new_site_domain[unique_site_domains$avg_clicks < q9 & unique_site_domains$avg_clicks > q8] <- 4
unique_site_domains$new_site_domain[unique_site_domains$avg_clicks > q9] <- 5

df <- left_join(df, unique_site_domains[, c(6, 27)], by = 'site_domain')
rm(unique_site_domains)

df <- df %>%
  group_by(site_domain) %>%
  mutate(count = n())

summary(distinct_at(df, .vars = 'site_domain', .keep_all = T)$count)
df$new_site_domain[df$count <= 20] <- 'low_freq'

#do the same thing for site_category------------------------------


# Gets the average ad clicks for each unique site id
df <- df %>%
  group_by(site_category) %>%
  mutate(avg_clicks = sum(click)/n())

# Here, if the site has an average click in the quantile ranges below, they get put into groups 1-5
# Anything below the median is classified in group 0 (Around half the websites get zero clicks)
unique_site_category <- distinct_at(df, .vars = 'site_category', .keep_all = T)
unique_site_category$new_site_category <- 0

sort(unique_site_category$avg_clicks)
unique_site_category$new_site_category[unique_site_category$avg_clicks < 0.1 & unique_site_category$avg_clicks > 0.0] <- 1
unique_site_category$new_site_category[unique_site_category$avg_clicks < 0.2 & unique_site_category$avg_clicks > 0.1] <- 2
unique_site_category$new_site_category[unique_site_category$avg_clicks < 0.4 & unique_site_category$avg_clicks > 0.2] <- 3
unique_site_category$new_site_category[unique_site_category$avg_clicks > 0.4] <- 4

df <- left_join(df, unique_site_category[, c(7, 28)], by = 'site_category')
rm(unique_site_category)

df <- df %>%
  group_by(site_category) %>%
  mutate(count = n())

summary(distinct_at(df, .vars = 'site_category', .keep_all = T)$count)
df$new_site_category[df$count <= 50] <- 'low_freq'

#do the same thing for app_id------------------------------


# Gets the average ad clicks for each unique site id
df <- df %>%
  group_by(app_id) %>%
  mutate(avg_clicks = sum(click)/n())

# Here, if the site has an average click in the quantile ranges below, they get put into groups 1-5
# Anything below the median is classified in group 0 (Around half the websites get zero clicks)
unique_app_id <- distinct_at(df, .vars = 'app_id', .keep_all = T)
unique_app_id$new_app_id <- 0

q7 <- quantile(unique_app_id$avg_clicks, 0.7)
q8 <- quantile(unique_app_id$avg_clicks, 0.8)
q9 <- quantile(unique_app_id$avg_clicks, 0.9)

unique_app_id$new_app_id[unique_app_id$avg_clicks < q8 & unique_app_id$avg_clicks > q7] <- 1
unique_app_id$new_app_id[unique_app_id$avg_clicks < q9 & unique_app_id$avg_clicks > q8] <- 2
unique_app_id$new_app_id[unique_app_id$avg_clicks > q9] <- 3

df <- left_join(df, unique_app_id[, c(8, 29)], by = 'app_id')
rm(unique_app_id)

df <- df %>%
  group_by(app_id) %>%
  mutate(count = n())

# cutoff is between the median and 3rd quartile; we believe it is reasonable
summary(distinct_at(df, .vars = 'app_id', .keep_all = T)$count)
df$new_app_id[df$count <= 10] <- 'low_freq'


#do the same thing for app_domain------------------------------


# Gets the average ad clicks for each unique site id
df <- df %>%
  group_by(app_domain) %>%
  mutate(avg_clicks = sum(click)/n())

# Here, if the site has an average click in the quantile ranges below, they get put into groups 1-5
# Anything below the median is classified in group 0 (Around half the websites get zero clicks)
unique_app_domain <- distinct_at(df, .vars = 'app_domain', .keep_all = T)
unique_app_domain$new_app_domain <- 0

unique_app_domain$new_app_domain[unique_app_domain$avg_clicks < 0.1] <- 1
unique_app_domain$new_app_domain[unique_app_domain$avg_clicks < 0.25 & unique_app_domain$avg_clicks >= 0.1] <- 2
unique_app_domain$new_app_domain[unique_app_domain$avg_clicks < 0.5 & unique_app_domain$avg_clicks >= 0.25] <- 3
unique_app_domain$new_app_domain[unique_app_domain$avg_clicks < 1.0 & unique_app_domain$avg_clicks >= 0.5] <- 4
unique_app_domain$new_app_domain[unique_app_domain$avg_clicks == 1.0] <- 5

df <- left_join(df, unique_app_domain[, c(9, 30)], by = 'app_domain')
rm(unique_app_domain)

df <- df %>%
  group_by(app_domain) %>%
  mutate(count = n())

# between median and third quartile
summary(distinct_at(df, .vars = 'app_domain', .keep_all = T)$count)
df$new_app_domain[df$count <= 15] <- 'low_freq'


#do the same thing for app_category------------------------------


# Gets the average ad clicks for each unique site id
df <- df %>%
  group_by(app_category) %>%
  mutate(avg_clicks = sum(click)/n())

# Here, if the site has an average click in the quantile ranges below, they get put into groups 1-5
# Anything below the median is classified in group 0 (Around half the websites get zero clicks)
unique_app_category <- distinct_at(df, .vars = 'app_category', .keep_all = T)
unique_app_category$new_app_category <- 0

unique_app_category$new_app_category[unique_app_category$avg_clicks < 0.15 & unique_app_category$avg_clicks > 0.0] <- 1
unique_app_category$new_app_category[unique_app_category$avg_clicks < 0.25 & unique_app_category$avg_clicks > 0.15] <- 2
unique_app_category$new_app_category[unique_app_category$avg_clicks > 0.25] <- 3

df <- left_join(df, unique_app_category[, c(10, 31)], by = 'app_category')
rm(unique_app_category)

df <- df %>%
  group_by(app_category) %>%
  mutate(count = n())

# median for our sample is 154 instances, which is a lot; so we go in between the 1st quartile and median
summary(distinct_at(df, .vars = 'app_category', .keep_all = T)$count)
df$new_app_category[df$count <= 75] <- 'low_freq'


#do the same thing for device_id------------------------------


# Gets the average ad clicks for each unique site id
df <- df %>%
  group_by(device_id) %>%
  mutate(avg_clicks = sum(click)/n())

# Here, if the site has an average click in the quantile ranges below, they get put into groups 1-5
# Anything below the median is classified in group 0 (Around half the websites get zero clicks)
unique_device_id <- distinct_at(df, .vars = 'device_id', .keep_all = T)
unique_device_id$new_device_id <- 0

unique_device_id$new_device_id[unique_device_id$avg_clicks < 0.4 & unique_device_id$avg_clicks >= 0.2] <- 1
unique_device_id$new_device_id[unique_device_id$avg_clicks < 0.6 & unique_device_id$avg_clicks >= 0.4] <- 2
unique_device_id$new_device_id[unique_device_id$avg_clicks < 0.8 & unique_device_id$avg_clicks >= 0.6] <- 3
unique_device_id$new_device_id[unique_device_id$avg_clicks >= 0.8] <- 4

df <- left_join(df, unique_device_id[, c(11, 32)], by = 'device_id')
rm(unique_device_id)

df <- df %>%
  group_by(device_id) %>%
  mutate(count = n())

# After looking at the unique frequencies for each id, we feel that 30 is a reasonable threshold
summary(distinct_at(df, .vars = 'device_id', .keep_all = T)$count)
sort(unique(distinct_at(df, .vars = 'device_id', .keep_all = T)$count))
df$new_device_id[df$count <= 30] <- 'low_freq'

#do the same thing for device_ip------------------------------


# Gets the average ad clicks for each unique site id
df <- df %>%
  group_by(device_ip) %>%
  mutate(avg_clicks = sum(click)/n())

# Here, if the site has an average click in the quantile ranges below, they get put into groups 1-5
# Anything below the median is classified in group 0 (Around half the websites get zero clicks)
unique_device_ip <- distinct_at(df, .vars = 'device_ip', .keep_all = T)
unique_device_ip$new_device_ip <- 0

unique_device_ip$new_device_ip[unique_device_ip$avg_clicks < 0.4 & unique_device_ip$avg_clicks >= 0.2] <- 1
unique_device_ip$new_device_ip[unique_device_ip$avg_clicks < 0.6 & unique_device_ip$avg_clicks >= 0.4] <- 2
unique_device_ip$new_device_ip[unique_device_ip$avg_clicks < 0.8 & unique_device_ip$avg_clicks >= 0.6] <- 3
unique_device_ip$new_device_ip[unique_device_ip$avg_clicks >= 0.8] <- 4

df <- left_join(df, unique_device_ip[, c(12, 33)], by = 'device_ip')
rm(unique_device_ip)

df <- df %>%
  group_by(device_ip) %>%
  mutate(count = n())

# The third quartile is still 1, we make low frequency below 50 instances here
summary(distinct_at(df, .vars = 'device_ip', .keep_all = T)$count)
sort(unique(distinct_at(df, .vars = 'device_ip', .keep_all = T)$count))
df$new_device_ip[df$count <= 50] <- 'low_freq'

#do the same thing for device_model------------------------------


# Gets the average ad clicks for each unique site id
df <- df %>%
  group_by(device_model) %>%
  mutate(avg_clicks = sum(click)/n())

# Here, if the site has an average click in the quantile ranges below, they get put into groups 1-5
# Anything below the median is classified in group 0 (Around half the websites get zero clicks)
unique_device_model <- distinct_at(df, .vars = 'device_model', .keep_all = T)
unique_device_model$new_device_model <- 0

q4 <- quantile(unique_device_model$avg_clicks, 0.4)
q5 <- quantile(unique_device_model$avg_clicks, 0.5)
q6 <- quantile(unique_device_model$avg_clicks, 0.6)
q7 <- quantile(unique_device_model$avg_clicks, 0.7)
q8 <- quantile(unique_device_model$avg_clicks, 0.8)
q9 <- quantile(unique_device_model$avg_clicks, 0.9)

unique_device_model$new_device_model[unique_device_model$avg_clicks < q5 & unique_device_model$avg_clicks >= q4] <- 1
unique_device_model$new_device_model[unique_device_model$avg_clicks < q6 & unique_device_model$avg_clicks >= q5] <- 2
unique_device_model$new_device_model[unique_device_model$avg_clicks < q7 & unique_device_model$avg_clicks >= q6] <- 3
unique_device_model$new_device_model[unique_device_model$avg_clicks < q8 & unique_device_model$avg_clicks >= q7] <- 4
unique_device_model$new_device_model[unique_device_model$avg_clicks < q9 & unique_device_model$avg_clicks >= q8] <- 5
unique_device_model$new_device_model[unique_device_model$avg_clicks > q9] <- 6

df <- left_join(df, unique_device_model[, c(13, 34)], by = 'device_model')
rm(unique_device_model)

df <- df %>%
  group_by(device_model) %>%
  mutate(count = n())

# Here we make the cutoff value less than 10, which is at the median
summary(distinct_at(df, .vars = 'device_model', .keep_all = T)$count)
sort(unique(distinct_at(df, .vars = 'device_ip', .keep_all = T)$count))
df$new_device_model[df$count < 10] <- 'low_freq'

# At this point, all of the categorical string columns have been reduced to a few groups
# We still need to check every other numeric categorical variables, to see if they have too many categories

length(unique(df$C1)) # good
length(unique(df$banner_pos)) # good
length(unique(df$device_type)) # good
length(unique(df$device_conn_type)) # good
length(unique(df$C14)) # bad (ugh)
length(unique(df$C15)) # good
length(unique(df$C16)) # good
length(unique(df$C17)) # bad (ugh)
length(unique(df$C18)) # good
length(unique(df$C19)) # bad (ugh)
length(unique(df$C20)) # bad (ugh)
length(unique(df$C21)) # bad (ugh)

#do the same thing for C14------------------------------


# Gets the average ad clicks for each unique site id
df <- df %>%
  group_by(C14) %>%
  mutate(avg_clicks = sum(click)/n())

# Here, if the site has an average click in the quantile ranges below, they get put into groups 1-5
# Anything below the median is classified in group 0 (Around half the websites get zero clicks)
unique_C14 <- distinct_at(df, .vars = 'C14', .keep_all = T)
unique_C14$new_C14 <- 0

q3 <- quantile(unique_C14$avg_clicks, 0.3)
q4 <- quantile(unique_C14$avg_clicks, 0.4)
q5 <- quantile(unique_C14$avg_clicks, 0.5)
q6 <- quantile(unique_C14$avg_clicks, 0.6)
q7 <- quantile(unique_C14$avg_clicks, 0.7)
q8 <- quantile(unique_C14$avg_clicks, 0.8)
q9 <- quantile(unique_C14$avg_clicks, 0.9)

unique_C14$new_C14[unique_C14$avg_clicks < q4 & unique_C14$avg_clicks >= q3] <- 1
unique_C14$new_C14[unique_C14$avg_clicks < q5 & unique_C14$avg_clicks >= q4] <- 2
unique_C14$new_C14[unique_C14$avg_clicks < q6 & unique_C14$avg_clicks >= q5] <- 3
unique_C14$new_C14[unique_C14$avg_clicks < q7 & unique_C14$avg_clicks >= q6] <- 4
unique_C14$new_C14[unique_C14$avg_clicks < q8 & unique_C14$avg_clicks >= q7] <- 5
unique_C14$new_C14[unique_C14$avg_clicks < q9 & unique_C14$avg_clicks >= q8] <- 6
unique_C14$new_C14[unique_C14$avg_clicks > q9] <- 7

df <- left_join(df, unique_C14[, c(16, 35)], by = 'C14')
rm(unique_C14)

df <- df %>%
  group_by(C14) %>%
  mutate(count = n())

# Here we make the cutoff value less than 40, which is at the median
summary(distinct_at(df, .vars = 'C14', .keep_all = T)$count)
sort(unique(distinct_at(df, .vars = 'C14', .keep_all = T)$count))
df$new_C14[df$count < 40] <- 'low_freq'


#do the same thing for C17------------------------------


# Gets the average ad clicks for each unique site id
df <- df %>%
  group_by(C17) %>%
  mutate(avg_clicks = sum(click)/n())

# Here, if the site has an average click in the quantile ranges below, they get put into groups 1-5
# Anything below the median is classified in group 0 (Around half the websites get zero clicks)
unique_C17 <- distinct_at(df, .vars = 'C17', .keep_all = T)
unique_C17$new_C17 <- 0

q2.5 <- quantile(unique_C17$avg_clicks, 0.25)
q5 <- quantile(unique_C17$avg_clicks, 0.5)
q7.5 <- quantile(unique_C17$avg_clicks, 0.75)

unique_C17$new_C17[unique_C17$avg_clicks < q5 & unique_C17$avg_clicks >= q2.5] <- 1
unique_C17$new_C17[unique_C17$avg_clicks < q7.5 & unique_C17$avg_clicks >= q5] <- 2
unique_C17$new_C17[unique_C17$avg_clicks >= q7.5] <- 3

df <- left_join(df, unique_C17[, c(19, 36)], by = 'C17')
rm(unique_C17)

df <- df %>%
  group_by(C17) %>%
  mutate(count = n())

# Here we make it 50 which is about half of the first quartile and still pretty low frequency comparatively
summary(distinct_at(df, .vars = 'C17', .keep_all = T)$count)
sort(unique(distinct_at(df, .vars = 'C17', .keep_all = T)$count))
df$new_C17[df$count < 50] <- 'low_freq'


#do the same thing for C19------------------------------


# Gets the average ad clicks for each unique site id
df <- df %>%
  group_by(C19) %>%
  mutate(avg_clicks = sum(click)/n())

# Here, if the site has an average click in the quantile ranges below, they get put into groups 1-5
# Anything below the median is classified in group 0 (Around half the websites get zero clicks)
unique_C19 <- distinct_at(df, .vars = 'C19', .keep_all = T)
unique_C19$new_C19 <- 0

q2.5 <- quantile(unique_C19$avg_clicks, 0.25)
q5 <- quantile(unique_C19$avg_clicks, 0.5)
q7.5 <- quantile(unique_C19$avg_clicks, 0.75)

unique_C19$new_C19[unique_C19$avg_clicks < q5 & unique_C19$avg_clicks >= q2.5] <- 1
unique_C19$new_C19[unique_C19$avg_clicks < q7.5 & unique_C19$avg_clicks >= q5] <- 2
unique_C19$new_C19[unique_C19$avg_clicks >= q7.5] <- 3

df <- left_join(df, unique_C19[, c(21, 37)], by = 'C19')
rm(unique_C19)

df <- df %>%
  group_by(C19) %>%
  mutate(count = n())

# Here we make it 50 for the same reasons as C17
summary(distinct_at(df, .vars = 'C19', .keep_all = T)$count)
sort(unique(distinct_at(df, .vars = 'C19', .keep_all = T)$count))
df$new_C19[df$count < 50] <- 'low_freq'


#do the same thing for C20------------------------------


# Gets the average ad clicks for each unique site id
df <- df %>%
  group_by(C20) %>%
  mutate(avg_clicks = sum(click)/n())

# Here, if the site has an average click in the quantile ranges below, they get put into groups 1-5
# Anything below the median is classified in group 0 (Around half the websites get zero clicks)
unique_C20 <- distinct_at(df, .vars = 'C20', .keep_all = T)
unique_C20$new_C20 <- 0

q2.5 <- quantile(unique_C20$avg_clicks, 0.25)
q5 <- quantile(unique_C20$avg_clicks, 0.5)
q7.5 <- quantile(unique_C20$avg_clicks, 0.75)

unique_C20$new_C20[unique_C20$avg_clicks < q5 & unique_C20$avg_clicks >= q2.5] <- 1
unique_C20$new_C20[unique_C20$avg_clicks < q7.5 & unique_C20$avg_clicks >= q5] <- 2
unique_C20$new_C20[unique_C20$avg_clicks >= q7.5] <- 3

df <- left_join(df, unique_C20[, c(22, 38)], by = 'C20')
rm(unique_C20)

df <- df %>%
  group_by(C20) %>%
  mutate(count = n())

# Here we make it 50 for the same reasons as C17
summary(distinct_at(df, .vars = 'C20', .keep_all = T)$count)
sort(unique(distinct_at(df, .vars = 'C20', .keep_all = T)$count))
df$new_C20[df$count < 50] <- 'low_freq'


#do the same thing for C21------------------------------


# Gets the average ad clicks for each unique site id
df <- df %>%
  group_by(C21) %>%
  mutate(avg_clicks = sum(click)/n())

# Here, if the site has an average click in the quantile ranges below, they get put into groups 1-5
# Anything below the median is classified in group 0 (Around half the websites get zero clicks)
unique_C21 <- distinct_at(df, .vars = 'C21', .keep_all = T)
unique_C21$new_C21 <- 0

q2.5 <- quantile(unique_C21$avg_clicks, 0.25)
q5 <- quantile(unique_C21$avg_clicks, 0.5)
q7.5 <- quantile(unique_C21$avg_clicks, 0.75)

unique_C21$new_C21[unique_C21$avg_clicks < q5 & unique_C21$avg_clicks >= q2.5] <- 1
unique_C21$new_C21[unique_C21$avg_clicks < q7.5 & unique_C21$avg_clicks >= q5] <- 2
unique_C21$new_C21[unique_C21$avg_clicks >= q7.5] <- 3

df <- left_join(df, unique_C21[, c(23, 39)], by = 'C21')
rm(unique_C21)

df <- df %>%
  group_by(C21) %>%
  mutate(count = n())

# Here we make it 300 because the count for this variable is quite high
summary(distinct_at(df, .vars = 'C21', .keep_all = T)$count)
sort(unique(distinct_at(df, .vars = 'C21', .keep_all = T)$count))
df$new_C21[df$count < 300] <- 'low_freq'


### Importing validation data and transforming the data into our new encodings
valdata <- fread('ProjectValidationData.csv')
valdata <- valdata[, -1]
colnames(valdata) <- names[-1]

# This will take a lot of joining
newValData <- left_join(valdata, distinct_at(df[, c(5, 25)], .vars = 'site_id', .keep_all = T), by = 'site_id')
newValData <- left_join(newValData, distinct_at(df[, c(6, 27)], .vars = 'site_domain', .keep_all = T), by = 'site_domain')
newValData <- left_join(newValData, distinct_at(df[, c(7, 28)], .vars = 'site_category', .keep_all = T), by = 'site_category')
newValData <- left_join(newValData, distinct_at(df[, c(8, 29)], .vars = 'app_id', .keep_all = T), by = 'app_id')
newValData <- left_join(newValData, distinct_at(df[, c(9, 30)], .vars = 'app_domain', .keep_all = T), by = 'app_domain')
newValData <- left_join(newValData, distinct_at(df[, c(10, 31)], .vars = 'app_category', .keep_all = T), by = 'app_category')
newValData <- left_join(newValData, distinct_at(df[, c(11, 32)], .vars = 'device_id', .keep_all = T), by = 'device_id')
newValData <- left_join(newValData, distinct_at(df[, c(12, 33)], .vars = 'device_ip', .keep_all = T), by = 'device_ip')
newValData <- left_join(newValData, distinct_at(df[, c(13, 34)], .vars = 'device_model', .keep_all = T), by = 'device_model')
newValData <- left_join(newValData, distinct_at(df[, c(16, 35)], .vars = 'C14', .keep_all = T), by = 'C14')
newValData <- left_join(newValData, distinct_at(df[, c(19, 36)], .vars = 'C17', .keep_all = T), by = 'C17')
newValData <- left_join(newValData, distinct_at(df[, c(21, 37)], .vars = 'C19', .keep_all = T), by = 'C19')
newValData <- left_join(newValData, distinct_at(df[, c(22, 38)], .vars = 'C20', .keep_all = T), by = 'C20')
newValData <- left_join(newValData, distinct_at(df[, c(23, 39)], .vars = 'C21', .keep_all = T), by = 'C21')

# We want to inspect how many NAs were generated from the join; overall this looks pretty good
# Except for the device id and ip, which was expected
sapply(newValData,FUN=function(x) { sum(is.na(x))})

newValData$new_device_id[is.na(newValData$new_device_id)] <- 'low_freq'
newValData$new_site_id[is.na(newValData$new_site_id)] <- 'low_freq'
newValData$new_site_domain[is.na(newValData$new_site_domain)] <- 'low_freq'
newValData$new_site_category[is.na(newValData$new_site_category)] <- 'low_freq'
newValData$new_app_id[is.na(newValData$new_app_id)] <- 'low_freq'
newValData$new_app_domain[is.na(newValData$new_app_domain)] <- 'low_freq'
newValData$new_app_category[is.na(newValData$new_app_category)] <- 'low_freq'
newValData$new_device_ip[is.na(newValData$new_device_ip)] <- 'low_freq'
newValData$new_device_model[is.na(newValData$new_device_model)] <- 'low_freq'
newValData$new_C14[is.na(newValData$new_C14)] <- 'low_freq'
newValData$new_C17[is.na(newValData$new_C17)] <- 'low_freq'
newValData$new_C19[is.na(newValData$new_C19)] <- 'low_freq'
newValData$new_C20[is.na(newValData$new_C20)] <- 'low_freq'
newValData$new_C21[is.na(newValData$new_C21)] <- 'low_freq'

newValData$hour <- as.character(newValData$hour)
newValData$hour <- substr(newValData$hour, 7, 8)

#write.csv(df, 'cleanedTraining.csv')
#df <- fread('cleanedTraining.csv')
#df <- df[, -1]

### Modeling!! 

# First, we are going to start with a random forest on the original data
# We tried to install tree package in AWS but it could not be downloaded
# We will try to run a basic decision tree on our own computer later

# We keep the original columns of the data, and get rid of the string columns as well
tree_df <- df[, c(1:4, 14:23)]
tree_val <- valdata[, c(1:4, 14:23)]
tree_df$click <- factor(tree_df$click)
tree_val$click <- factor(tree_val$click)

# Makes the formula
vars <- names(tree_df)
BigFM <- paste(vars[1], "~", paste(vars[2:14], collapse = '+'), sep = ' ')
BigFM <- formula(BigFM)

# Grows the forest
Sys.time()
out <- randomForest(BigFM,data=tree_df,ntree=25)
Sys.time()

# Make predictions
Yhat <- predict(out, newdata = tree_val, type = 'prob')

LogLoss <- function(y_pred, y_true) {
  eps <- 1e-15
  y_pred <- pmax(pmin(y_pred, 1 - eps), eps)
  LogLoss <- -mean(y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred))
  return(LogLoss)
}

LogLoss(Yhat[, 2], as.numeric(tree_val$click) - 1)

rm(tree_df)
rm(tree_val)
rm(valdata)

# 4.231

# Random forest on the new data

# keeps the numeric encodings of the string variables
cleanedTraining <- df[, c(1:4, 14, 15, 17, 18, 20, 25, 27:39)]
rm(df)

# If the forest is yielding errors, try running these lines, they might fix some problems
# cleanedTraining$hour <- str_pad(cleanedTraining$hour, 2, pad = '0')
# levels(cleanedValidation$device_type) <- levels(cleanedTraining$device_type)

cleanedValidation <- newValData[, c(1:4, 14, 15, 17, 18, 20, 24:37)]

vars <- names(cleanedTraining)
BigFM <- paste(vars[1], "~", paste(vars[2:23], collapse = '+'), sep = ' ')
BigFM <- formula(BigFM)

cleanedTraining <- sapply(cleanedTraining,FUN=function(x) { as.factor(x)})
cleanedValidation <- sapply(cleanedValidation,FUN=function(x) { as.factor(x)})
cleanedTraining <- as.data.frame(cleanedTraining)
cleanedValidation <- as.data.frame(cleanedValidation)

str(cleanedTraining)
str(cleanedValidation)

Sys.time()
out <- randomForest(BigFM,data=cleanedTraining,ntree=25)
Sys.time()

Yhat <- predict(out, newdata = cleanedValidation, type = 'prob')
LogLoss(Yhat[, 2], as.numeric(cleanedValidation$click)-1)

# 4.05

# Next we turn our factors into dummies
training_with_dummies <- model.matrix(click ~ ., cleanedTraining)















library(tidyverse) 
library(data.table)
library(rstudioapi)
library(skimr)
library(inspectdf)
library(mice)
library(plotly)
library(highcharter)
library(recipes) 
library(caret) 
library(purrr) 
library(graphics) 
library(Hmisc) 
library(glue)
library(h2o)



##1. Add ggplot2::mpg dataset
raw <- ggplot2::mpg
raw


## 2. Make data ready for analysis doing preprocessing techniques. 

raw %>% inspect_na()

# Splitting the data into Categorical and Numerical datas

df_car <- raw %>% select_if(is.character)

df_car %>% inspect_na()


df_num <- raw %>% select_if(is.numeric)


df_num %>% inspect_na()

# One hot Encoding

df_car <- dummyVars(" ~ .", data = df_car) %>% 
  predict(newdata = df_car) %>% 
  as.data.frame()

# Combining the encoded data with numerical data to make up a final dataset to create a model with

df <- cbind(df_num, df_car)


names(df) %>% str_replace_all(' ', '_') %>%
              str_replace_all("\\(","_") %>% 
              str_replace_all("\\)","") ->names(df)

df



# Multicolliniearity


target <- 'cyl'
features <- df %>% select(-cyl,) %>% names()



f <- as.formula(paste(target, paste(features, collapse = " + "), sep = " ~ "))
glm <- glm(f,data=df)


glm %>% summary()


## Getting rid of the features where coef is equal to na
coef_na <- attributes(alias(glm)$Complete)$dimnames[[1]]

features <- features[!features %in% coef_na]



f <- as.formula(paste(target, paste(features, collapse = " + "), sep = " ~ "))
glm <- glm(f,data=df)


glm %>% summary()



## Getting rid of the features with multicollinearity more than 3
                  

while(glm %>% faraway::vif() %>% sort(decreasing = T) %>% .[1] >= 3){
  afterVIF <- glm %>% faraway::vif() %>% sort(decreasing = T) %>% .[-1] %>% names()
  f <- as.formula(paste(target, paste(afterVIF, collapse = " + "), sep = " ~ "))
  glm <- glm(f, data = df)
}


features <- glm %>% faraway::vif() %>% sort(decreasing = T) %>% names()


df <- df %>% select(cyl,features)

## Scaling the data

df %>% glimpse()


df[,-1] <- df[,-1] %>% scale() %>% as.data.frame()


## Modeling


h2o.init()


h2o_df <- df %>% as.h2o()

# Splitting the data ----

h2o_df <- h2o_df %>% h2o.splitFrame(ratios = 0.8, seed = 123)

#h2o_df_split <- 

train <- h2o_df[[1]]

test <- h2o_df[[2]]


target <- 'cyl'

features <- df %>% select(-cyl) %>% names()

##=====Fitting h20 model

model <- h2o.glm(
  x = features, y = target,
  training_frame = train,
  validation_frame = test,
  nfolds = 10, seed = 123,
  lambda = 0, compute_p_values = T)

model@model$coefficients_table %>%
  as.data.frame() %>%
  dplyr::select(names,p_value) %>%
  mutate(p_value = round(p_value,3)) %>%
  .[-1,] %>%
  arrange(desc(p_value))



## Choosing only values where p_value is higher than 0.05 for statistical significance


while(model@model$coefficients_table %>%
      as.data.frame() %>%
      dplyr::select(names,p_value) %>%
      mutate(p_value = round(p_value,3)) %>%
      .[-1,] %>%
      arrange(desc(p_value)) %>%
      .[1,2] > 0.05) {
  model@model$coefficients_table %>%
    as.data.frame() %>%
    dplyr::select(names,p_value) %>%
    mutate(p_value = round(p_value,3)) %>%
    filter(!is.nan(p_value)) %>%
    .[-1,] %>%
    arrange(desc(p_value)) %>%
    .[1,1] -> v
  features <- features[features!=v]
  
  train_h2o <- train %>% as.data.frame() %>% select(target, features) %>% as.h2o()
  test_h2o <- test %>% as.data.frame() %>% select(target, features) %>% as.h2o()
  
  model <- h2o.glm(
    x = features, y = target,
    training_frame = train,
    validation_frame = test,
    nfolds = 10, seed = 123,
    lambda = 0, compute_p_values = T)
}


## Predicting the test results

y_pred <- model %>% h2o.predict(newdata = test) %>% as.data.frame()
y_pred$predict



## Model Evaluation


test_set <- test %>% as.data.frame()
residuals = test_set$cyl - y_pred$predict

residuals


# Calculate RMSE (Root Mean Square Error) ----
RMSE = sqrt(mean(residuals^2))

RMSE

# Calculate Adjusted R2 (R Squared) ----
y_test_mean = mean(test_set$cyl)

y_test_mean

tss = sum((test_set$cyl - y_test_mean)^2) #total sum of squares
rss = sum(residuals^2) #residual sum of squares

R2 =1-(rss/tss)

R2

n <- test_set %>% nrow() #sample size
k <- features %>% length() #number of independent variables
Adjusted_R2 = 1-(1-R2)*((n-1)/(n-k-1))

Adjusted_R2


tibble(RMSE = round(RMSE,1),
       R2, Adjusted_R2)

library(dplyr)
library(tidyr)
library(xgboost)
library(glmnet)
library(doParallel)
library(foreach)


link_inv <- function(x){
  return(1/(1 + exp(-x)))
}

link_inv_der <- function(x){
  return(link_inv(x) * link_inv(-x))
}



eICU = read.csv("alldata_cleaned.csv")[,-1]
eICU = eICU[eICU$ethnicity != "Other/Unknown", ]
eICU = eICU[eICU$apachedxgroup != "Other", ]
eICU$age_group = as.factor(eICU$age_group)


eICU = eICU %>% mutate(value = 1)  %>% spread(apachedxgroup, value,  fill = 0 )



eICU.source = eICU |> filter(ethnicity == "Caucasian") |> select(-ethnicity)
x_1 = model.matrix(actualhospitalmortality ~ .-1, data = eICU.source)
y_1 = as.numeric(eICU.source$actualhospitalmortality == "EXPIRED")
d <- xgb.DMatrix(data = x_1, label = y_1)


# Parameters
params <- list(
  booster = "gbtree",
  objective = "binary:logistic",
  eval_metric = "auc",
  max_depth = 4,
  eta = 0.1,
  subsample = 0.8,
  colsample_bytree = 0.8
)

# Cross-validation
cv.nrounds <- 500
cv.nfold <- 5

cv <- xgb.cv(
  params = params,
  data = d,
  nrounds = cv.nrounds,
  nfold = cv.nfold,
  showsd = TRUE,
  stratified = TRUE,
  print_every_n = 20,
  early_stopping_rounds = 20,
  maximize = TRUE
)

# Best number of rounds by CV AUC
best_nrounds <- cv$best_iteration
source_model <- xgb.train(params = params, data = d, nrounds = best_nrounds)
rm(cv)
rm(params)

saveRDS(source_model, file = "source_model.rds")


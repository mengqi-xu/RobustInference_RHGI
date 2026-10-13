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


eICU.target = eICU |> filter(ethnicity == "Asian") |> select(-ethnicity)
x_0 = model.matrix(actualhospitalmortality ~ .-1, data = eICU.target)
y_0 = as.numeric(eICU.target$actualhospitalmortality == "EXPIRED")

x = rbind(x_0, x_1)
y = c(y_0, y_1)
n0 = length(y_0)
n1 = length(y_1)
S = c(rep(0, n0), rep(1, n1))

d <- xgb.DMatrix(data = x, label = S)


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
propensity_model <- xgb.train(params = params, data = d, nrounds = best_nrounds)
rm(cv)
rm(params)

d <- xgb.DMatrix(data = x, label = y)
offset = predict(source_model, d)
offset = log(offset / (1 - offset))

pred_prop = predict(propensity_model, d)

predictor_0 = cv.glmnet(x_0, y_0, offset = offset[1:n0], family = binomial())


# estimate from the adjustment model
theta_hat_1 = coef(predictor_0, s = "lambda.min")[2]

## unpenalized estimator (benchmark): glm on target data with source offset
fit_unpen = suppressWarnings(glm(y_0 ~ x_0, offset = offset[1:n0], family = binomial()))
cf = summary(fit_unpen)$coefficients
## glm drops aliased columns, so match coefficients by name (NA if dropped)
idx = match(paste0("x_0", colnames(x)), row.names(cf))
theta_unpen = cf[idx, 1]
sd_unpen = cf[idx, 2]

## prediction of the adjustment
l_0 = predict(predictor_0, newx = x, type = "link", s = "lambda.min", newoffset = y * 0)

## variable want to estimate
theta_hats = rep(0, dim(x)[2])
sd_hats = rep(0, length(theta_hats))
theta_hats_biased = rep(0, length(theta_hats))


totalCores = detectCores()
cluster <- makeCluster(totalCores[1]-1)
registerDoParallel(cluster)

theta_hat = foreach(i = 1:dim(x)[2], .packages = "glmnet") %dopar%{
  ## required k and Lambda for neyman orthogonalization
  k = (pred_prop * link_inv_der(offset)) / (pred_prop * link_inv_der(offset) + (1 - pred_prop) * link_inv_der(offset + l_0))
  LAMBDA_model = cv.glmnet(x[, -i], x[, i], weights = k * (1 - pred_prop) * link_inv_der(offset + l_0))

  # initialization for iteration
  theta_init = theta_hat_1
  err = 1.0
  ITER = 0

  while(err > 1e-7 && ITER < 1000){
    w = x[, i] - predict(LAMBDA_model, newx = x[, -i], type = "link", s = "lambda.min")
    g = (1 - S) * k * (link_inv(offset + (theta_init - theta_hat_1) * x[, i] + l_0) - y) * w - S * (1 - k) * (link_inv(offset) - y) * w
    h = (1 - S) * k * link_inv_der(offset + (theta_init - theta_hat_1) * x[, i] + l_0) * (w * x[, i])
    theta_upd = theta_init - max(-10, min(sum(g) / sum(h), 10))
    err = abs(theta_upd - theta_init)
    ITER = ITER + 1
    theta_init = theta_upd
  }
  theta_hat_2 = theta_upd
  sd_2 = sd(g/mean(h))/sqrt(n0+n1)

  c(theta_hat_1, theta_hat_2, sd_2)
}
stopCluster(cluster)

theta_hat = matrix(unlist(theta_hat), 3, dim(x)[2])
results = data.frame(coef.biased = theta_hat[1,], coef_NO = theta_hat[2,], sd_NO = theta_hat[3,])
results$lower_bound = results$coef_NO - 1.96 * results$sd_NO
results$upper_bound = results$coef_NO + 1.96 * results$sd_NO
results$pvalue = 1 - pnorm(abs(results$coef_NO) / results$sd_NO)
results$coef_unpen = theta_unpen
results$sd_unpen = sd_unpen
results$lower_bound_unpen = results$coef_unpen - 1.96 * results$sd_unpen
results$upper_bound_unpen = results$coef_unpen + 1.96 * results$sd_unpen
results$pvalue_unpen = 1 - pnorm(abs(results$coef_unpen) / results$sd_unpen)
row.names(results) = colnames(x)

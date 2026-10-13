library(xgboost)
library(glmnet)
library(doParallel)
library(foreach)
library(latex2exp)
library(tidyr)
library(dplyr)
library(torch)
library(pROC)

# ── Cluster setup ───────────────────────────────────────────────────────────
ncores  <- as.integer(Sys.getenv("NCORES", Sys.getenv("SLURM_CPUS_PER_TASK", "2")))
cluster <- makeCluster(ncores)
registerDoParallel(cluster)
invisible(clusterEvalQ(cluster, {
  Sys.setenv(OMP_NUM_THREADS = "1")
  library(torch); torch_set_num_threads(1)
  NULL
}))
cat(sprintf("Using %d workers\n", ncores))

args <- commandArgs(trailingOnly = TRUE)

link_inv     <- function(x) 1 / (1 + exp(-x))
link_inv_der <- function(x) link_inv(x) * link_inv(-x)

get_data <- function(n, theta, d = 20) {
  x <- matrix(rnorm(n * d), n, d)
  x[, 3] <- rbinom(n, 1, 0.5)
  x[, 4] <- rbinom(n, 1, 0.4)
  x[, 5] <- rbinom(n, 1, 0.3)
  x[, 11] <- rbinom(n, 1, 0.5)
  x[, 13] <- sample(0:2, n, replace = TRUE)
  
  l <- log(0.1 / 0.9) - 3 * rowMeans(x[, 1:10]) + theta * (x[, 1] + x[, 2])
  p <- link_inv(l)
  y <- rbinom(n, 1, prob = p)
  list(x = x, y = y)
}


# ── Single-layer NN ──────
train_nn_source <- function(x_train, y_train, lr = 0.01, epochs = 50) {
  p <- ncol(x_train)
  loss_fn <- torch::nn_bce_loss()
  
  model <- torch::nn_sequential(torch::nn_linear(p, 100),
                                torch::nn_relu(),
                                torch::nn_linear(100, 1),
                                torch::nn_sigmoid())
  opt <- torch::optim_adam(model$parameters, lr = lr)
  x_t <- torch::torch_tensor(x_train, dtype = torch::torch_float())
  y_t <- torch::torch_tensor(y_train, dtype = torch::torch_float())$unsqueeze(2)
  
  for (epoch in seq_len(epochs)) {
    opt$zero_grad()
    loss <- loss_fn(model(x_t), y_t)
    loss$backward()
    opt$step()
  }
  return(model)
}


nn_predict_logodds <- function(model, x_new) {
  model$eval()
  x_t <- torch::torch_tensor(x_new, dtype = torch::torch_float())
  torch::with_no_grad({
    prob <- as.numeric(model(x_t))
  })
  prob <- pmin(pmax(prob, 1e-6), 1 - 1e-6)
  return(log(prob / (1 - prob)))
}

# ── Simulation settings ────────────────────────────────────────────────────────
theta    <- 0.5
N        <- 30000
ratios   <- if (length(args) >= 1) as.numeric(args[1]) else c(0.4, 0.2, 0.1)
max_iter <- if (length(args) >= 2) as.integer(args[2]) else 1000
d = 20
all_results <- list()

for (ratio in ratios) {
  
  nt <- round(ratio * N)
  ns <- round((1 - ratio) * N)
  cat(sprintf("\n>>> ratio of target = %.2f  (nt = %d)\n", ratio, nt))
  
  res <- foreach(iter = 1:max_iter,
                 .packages = c("glmnet", "xgboost", "torch", "pROC")) %dopar% {
                   
                   # set.seed(1000 + ratio * 10 + iter * 100)
                   set.seed(iter * 1000)
                   
                   data_1 <- get_data(ns, theta = 0, d = 20);     x_1 <- data_1$x; y_1 <- data_1$y
                   data_0 <- get_data(nt, theta = theta, d = 20); x_0 <- data_0$x; y_0 <- data_0$y
                   
                   S <- c(rep(0, nt), rep(1, ns))
                   x <- rbind(x_0, x_1)
                   y <- c(y_0, y_1)
                   
                   # # ── XGB baseline ──────────────────────────────────────────────────────────
                   # d_xgb  <- xgboost::xgb.DMatrix(data = x_1, label = y_1)
                   # params <- list(booster = "gbtree", objective = "binary:logistic",
                   #                eval_metric = "auc", max_depth = 2, eta = 0.02,
                   #                subsample = 0.8, colsample_bytree = 1)
                   # cv     <- xgboost::xgb.cv(params = params, data = d_xgb, nrounds = 500,
                   #                           nfold = 5, showsd = FALSE, stratified = TRUE,
                   #                           print_every_n = 500, early_stopping_rounds = 20,
                   #                           maximize = TRUE, verbose = FALSE)
                   # mdl_xgb    <- xgboost::xgb.train(params = params, data = d_xgb,
                   #                                  nrounds = cv$best_iteration, verbose = 0)
                   # d_full     <- xgboost::xgb.DMatrix(data = x, label = y)
                   # prob_xgb   <- predict(mdl_xgb, d_full)
                   # offset_xgb <- log(prob_xgb / (1 - prob_xgb))
                   # 
                   # # LinearAdj XGB
                   # fit_xgb_adj  <- cv.glmnet(x_0, y_0, offset = offset_xgb[1:nt], family = binomial())
                   # theta_xgb_la <- coef(fit_xgb_adj, s = "lambda.min")[2]
                   # 
                   # # XGB Neyman Orthogonal
                   # l_0_xgb  <- predict(fit_xgb_adj, newx = x, type = "link",
                   #                     s = "lambda.min", newoffset = y * 0)
                   # prop_xgb <- cv.glmnet(x, S, family = binomial())
                   # pp_xgb   <- predict(prop_xgb, newx = x, type = "response", s = "lambda.min")
                   # k_xgb    <- (pp_xgb * link_inv_der(offset_xgb)) /
                   #   (pp_xgb * link_inv_der(offset_xgb) +
                   #      (1 - pp_xgb) * link_inv_der(offset_xgb + l_0_xgb))
                   # lam_xgb  <- cv.glmnet(x[, -1], x[, 1],
                   #                       weights = k_xgb * (1 - pp_xgb) *
                   #                         link_inv_der(offset_xgb + l_0_xgb))
                   # 
                   # ti <- theta_xgb_la; err <- 1; IT <- 0
                   # while (err > 1e-4 && IT < 1000) {
                   #   w  <- x[,1] - predict(lam_xgb, newx = x[,-1], type = "link", s = "lambda.min")
                   #   g  <- (1-S)*k_xgb*(link_inv(offset_xgb + (ti-theta_xgb_la)*x[,1] + l_0_xgb) - y)*w -
                   #     S*(1-k_xgb)*(link_inv(offset_xgb) - y)*w
                   #   h  <- (1-S)*k_xgb*link_inv_der(offset_xgb + (ti-theta_xgb_la)*x[,1] + l_0_xgb)*(w*x[,1]) #####
                   #   tu <- ti - max(-10, min(sum(g)/sum(h), 10))
                   #   err <- abs(tu - ti); IT <- IT + 1; ti <- tu
                   # }
                   # 
                   # 
                   # theta_xgb_no <- tu
                   # 
                   # g  <- (1-S)*k_xgb*(link_inv(offset_xgb + (ti-theta_xgb_la)*x[,1] + l_0_xgb) - y)*w -
                   #   S*(1-k_xgb)*(link_inv(offset_xgb) - y)*w
                   # h  <- (1-S)*k_xgb*link_inv_der(offset_xgb + (ti-theta_xgb_la)*x[,1] + l_0_xgb)*(w^2) #####
                   # 
                   # # sd_xgb_no    <- sd(g / mean(h)) / sqrt(ns + nt)
                   # sd_xgb_no    <- sqrt(var(g) / ((ns+nt) * mean(h)^2) )
                   
                   # ── XGB baseline ──────────────────────────────────────────────────────────
                   d_xgb  <- xgboost::xgb.DMatrix(data = x_1, label = y_1)
                   params <- list(booster = "gbtree", objective = "binary:logistic",
                                  eval_metric = "auc", max_depth = 2, eta = 0.01,
                                  subsample = 0.75, colsample_bytree = 0.75,
                                  nthread = 1)
                   cv     <- xgboost::xgb.cv(params = params, data = d_xgb, nrounds = 1000,
                                             nfold = 5, showsd = FALSE, stratified = TRUE,
                                             print_every_n = 500, early_stopping_rounds = 40,
                                             maximize = TRUE, verbose = FALSE)
                   # 兼容新旧版本 xgboost：1.x 用 cv$best_iteration，2.x/3.x 存在 cv$early_stop 里
                   best_it <- cv$best_iteration
                   if (is.null(best_it)) best_it <- cv$early_stop$best_iteration
                   if (is.null(best_it)) best_it <- nrow(cv$evaluation_log)
                   mdl_xgb    <- xgboost::xgb.train(params = params, data = d_xgb,
                                                    nrounds = best_it, verbose = 0)
                   d_full     <- xgboost::xgb.DMatrix(data = x, label = y)
                   prob_xgb   <- predict(mdl_xgb, d_full)
                   offset_xgb <- log(prob_xgb / (1 - prob_xgb))
                   
                   # LinearAdj XGB
                   fit_xgb_adj  <- cv.glmnet(x_0, y_0, offset = offset_xgb[1:nt], family = binomial())
                   theta_xgb_la <- coef(fit_xgb_adj, s = "lambda.min")[2]
                   
                   # Unpen XGB
                   fit_xgb_unpen <- suppressWarnings(glm(y_0 ~ x_0, offset = offset_xgb[1:nt], family = binomial()))
                   cf  <- summary(fit_xgb_unpen)$coefficients
                   theta_xgb_unpen = cf[2, 1]
                   sd_xgb_unpen = cf[2, 2]
                   
                   # XGB Neyman Orthogonal
                   l_0_xgb  <- predict(fit_xgb_adj, newx = x, type = "link",
                                       s = "lambda.min", newoffset = y * 0)
                   prop_xgb <- cv.glmnet(x, S, family = binomial())
                   pp_xgb   <- predict(prop_xgb, newx = x, type = "response", s = "lambda.min")
                   k_xgb    <- (pp_xgb * link_inv_der(offset_xgb)) /
                     (pp_xgb * link_inv_der(offset_xgb) +
                        (1 - pp_xgb) * link_inv_der(offset_xgb + l_0_xgb))
                   lam_xgb  <- cv.glmnet(x[, -1], x[, 1],
                                         weights = k_xgb * (1 - pp_xgb) *
                                           link_inv_der(offset_xgb + l_0_xgb))
                   
                   ti <- theta_xgb_la; err <- 1; IT <- 0
                   while (err > 1e-4 && IT < 1000) {
                     w  <- x[,1] - predict(lam_xgb, newx = x[,-1], type = "link", s = "lambda.min")
                     g  <- (1-S)*k_xgb*(link_inv(offset_xgb + (ti-theta_xgb_la)*x[,1] + l_0_xgb) - y)*w -
                       S*(1-k_xgb)*(link_inv(offset_xgb) - y)*w
                     h  <- (1-S)*k_xgb*link_inv_der(offset_xgb + (ti-theta_xgb_la)*x[,1] + l_0_xgb)*(w*x[,1])
                     tu <- ti - max(-10, min(sum(g)/sum(h), 10))
                     err <- abs(tu - ti); IT <- IT + 1; ti <- tu
                   }
                   theta_xgb_no <- tu
                   g  <- (1-S)*k_xgb*(link_inv(offset_xgb + (tu-theta_xgb_la)*x[,1] + l_0_xgb) - y)*w -
                     S*(1-k_xgb)*(link_inv(offset_xgb) - y)*w
                   h  <- (1-S)*k_xgb*link_inv_der(offset_xgb + (tu-theta_xgb_la)*x[,1] + l_0_xgb)*(w^2)
                   sd_xgb_no    <- sd(g) / (mean(h) * sqrt(ns + nt))
                   
                   # ── GLM baseline ──────────────────────────────────────────────────────────
                   mdl_glm    <- cv.glmnet(x_1, y_1, family = binomial())
                   prop_glm   <- cv.glmnet(x, S, family = binomial())
                   offset_glm <- predict(mdl_glm, newx = x, type = "link", s = "lambda.min")
                   pp_glm     <- predict(prop_glm, newx = x, type = "response", s = "lambda.min")
                   
                   # LinearAdj GLM
                   fit_glm_adj  <- cv.glmnet(x_0, y_0, offset = offset_glm[1:nt], family = binomial())
                   theta_glm_la <- coef(fit_glm_adj, s = "lambda.min")[2]
                   
                   # Unpen GLM
                   fit_glm_unpen <- suppressWarnings(glm(y_0 ~ x_0, offset = offset_glm[1:nt], family = binomial()))
                   cf  <- summary(fit_glm_unpen)$coefficients
                   theta_glm_unpen = cf[2, 1]
                   sd_glm_unpen = cf[2, 2]
                   
                   # GLM Neyman Orthogonal
                   l_0_glm <- predict(fit_glm_adj, newx = x, type = "link",
                                      s = "lambda.min", newoffset = y * 0)
                   k_glm   <- (pp_glm * link_inv_der(offset_glm)) /
                     (pp_glm * link_inv_der(offset_glm) +
                        (1 - pp_glm) * link_inv_der(offset_glm + l_0_glm))
                   lam_glm <- cv.glmnet(x[, -1], x[, 1],
                                        weights = k_glm * (1 - pp_glm) *
                                          link_inv_der(offset_glm + l_0_glm))
                   
                   ti <- theta_glm_la; err <- 1; IT <- 0
                   while (err > 1e-4 && IT < 1000) {
                     w  <- x[,1] - predict(lam_glm, newx = x[,-1], type = "link", s = "lambda.min")
                     g  <- (1-S)*k_glm*(link_inv(offset_glm + (ti-theta_glm_la)*x[,1] + l_0_glm) - y)*w -
                       S*(1-k_glm)*(link_inv(offset_glm) - y)*w
                     h  <- (1-S)*k_glm*link_inv_der(offset_glm + (ti-theta_glm_la)*x[,1] + l_0_glm)*(w* x[,1]) #*x
                     tu <- ti - max(-10, min(sum(g)/sum(h), 10))
                     err <- abs(tu - ti); IT <- IT + 1; ti <- tu
                   }
                   theta_glm_no <- tu
                   
                   g  <- (1-S)*k_glm*(link_inv(offset_glm + (ti-theta_glm_la)*x[,1] + l_0_glm) - y)*w -
                     S*(1-k_glm)*(link_inv(offset_glm) - y)*w
                   h  <- (1-S)*k_glm*link_inv_der(offset_glm + (ti-theta_glm_la)*x[,1] + l_0_glm)*(w^2) #*x
                   # sd_glm_no    <- sd(g / mean(h)) / sqrt(ns + nt)
                   sd_glm_no    <- sqrt(var(g) / ((ns+nt) * mean(h)^2) )
                   
                   # ── NN baseline ───────────────────────────────────────────────────────────
                   mdl_nn     <- train_nn_source(x_1, y_1)
                   offset_nn  <- nn_predict_logodds(mdl_nn, x)
                   prop_nn    <- cv.glmnet(x, S, family = binomial())
                   pp_nn      <- predict(prop_nn, newx = x, type = "response", s = "lambda.min")
                   
                   # LinearAdj NN
                   fit_nn_adj  <- cv.glmnet(x_0, y_0, offset = offset_nn[1:nt], family = binomial())
                   theta_nn_la <- coef(fit_nn_adj, s = "lambda.min")[2]
                   
                   # Unpen GLM
                   fit_nn_unpen <- suppressWarnings(glm(y_0 ~ x_0, offset = offset_nn[1:nt], family = binomial()))
                   cf  <- summary(fit_nn_unpen)$coefficients
                   theta_nn_unpen = cf[2, 1]
                   sd_nn_unpen = cf[2, 2]
                   
                   # NN Neyman Orthogonal
                   l_0_nn <- predict(fit_nn_adj, newx = x, type = "link",
                                     s = "lambda.min", newoffset = y * 0)
                   k_nn   <- (pp_nn * link_inv_der(offset_nn)) /
                     (pp_nn * link_inv_der(offset_nn) +
                        (1 - pp_nn) * link_inv_der(offset_nn + l_0_nn))
                   lam_nn <- cv.glmnet(x[, -1], x[, 1],
                                       weights = k_nn * (1 - pp_nn) *
                                         link_inv_der(offset_nn + l_0_nn))
                   
                   ti <- theta_nn_la; err <- 1; IT <- 0
                   while (err > 1e-4 && IT < 1000) {
                     w  <- x[,1] - predict(lam_nn, newx = x[,-1], type = "link", s = "lambda.min")
                     g  <- (1-S)*k_nn*(link_inv(offset_nn + (ti-theta_nn_la)*x[,1] + l_0_nn) - y)*w -
                       S*(1-k_nn)*(link_inv(offset_nn) - y)*w
                     h  <- (1-S)*k_nn*link_inv_der(offset_nn + (ti-theta_nn_la)*x[,1] + l_0_nn)*(w^2) #####
                     tu <- ti - max(-10, min(sum(g)/sum(h), 10))
                     err <- abs(tu - ti); IT <- IT + 1; ti <- tu
                   }
                   theta_nn_no <- tu
                   # sd_nn_no    <- sd(g / mean(h)) / sqrt(ns + nt)
                   sd_nn_no    <- sqrt(var(g) / ((ns+nt) * mean(h)^2) )
                   
                   # ── Coverage ────────────────────────────────────────────────────
                   cov_xgb <- as.numeric(abs(theta_xgb_no - theta) <= 1.96 * sd_xgb_no)
                   cov_glm <- as.numeric(abs(theta_glm_no - theta) <= 1.96 * sd_glm_no)
                   cov_nn  <- as.numeric(abs(theta_nn_no  - theta) <= 1.96 * sd_nn_no)
                   
                   cov_xgb_unpen <- as.numeric(abs(theta_xgb_unpen - theta) <= 1.96 * sd_xgb_unpen)
                   cov_glm_unpen <- as.numeric(abs(theta_glm_unpen - theta) <= 1.96 * sd_glm_unpen)
                   cov_nn_unpen  <- as.numeric(abs(theta_nn_unpen  - theta) <= 1.96 * sd_nn_unpen)
                   
                   c(theta_xgb_la, theta_xgb_unpen, theta_xgb_no, 
                     theta_glm_la, theta_glm_unpen, theta_glm_no,
                     theta_nn_la,  theta_nn_unpen, theta_nn_no,
                     sd_xgb_no, sd_glm_no, sd_nn_no,
                     sd_xgb_unpen, sd_glm_unpen, sd_nn_unpen,
                     cov_xgb, cov_glm, cov_nn,
                     cov_xgb_unpen, cov_glm_unpen, cov_nn_unpen)
                 }
  
  mat <- matrix(unlist(res), 21, max_iter)
  # row 1:  theta_xgb_la
  # row 2:  theta_xgb_unpen
  # row 3:  theta_xgb_no
  # row 4:  theta_glm_la
  # row 5:  theta_glm_unpen
  # row 6:  theta_glm_no
  # row 7:  theta_nn_la
  # row 8:  theta_nn_unpen
  # row 9:  theta_nn_no
  # row 10: sd_xgb_no
  # row 11: sd_glm_no
  # row 12: sd_nn_no
  # row 13: sd_xgb_unpen
  # row 14: sd_glm_unpen
  # row 15: sd_nn_unpen
  # row 16: cov_xgb_no
  # row 17: cov_glm_no
  # row 18: cov_nn_no
  # row 19: cov_xgb_unpen
  # row 20: cov_glm_unpen
  # row 21: cov_nn_unpen

  all_results[[as.character(ratio)]] <- data.frame(
    ratio     = ratio,
    nt        = nt,
    method    = rep(c("LinearAdj XGB", "LinearAdjUnpen XGB", "Debiased XGB",
                      "LinearAdj GLM", "LinearAdjUnpen GLM", "Debiased GLM",
                      "LinearAdj NN",  "LinearAdjUnpen NN",  "Debiased NN"), each = max_iter),
    theta_hat = c(mat[1,], mat[2,], mat[3,],
                  mat[4,], mat[5,], mat[6,],
                  mat[7,], mat[8,], mat[9,]),
    sd_est    = c(rep(NA, max_iter), mat[13,], mat[10,],
                  rep(NA, max_iter), mat[14,], mat[11,],
                  rep(NA, max_iter), mat[15,], mat[12,]),
    covered   = c(rep(NA, max_iter), mat[19,], mat[16,],
                  rep(NA, max_iter), mat[20,], mat[17,],
                  rep(NA, max_iter), mat[21,], mat[18,])
  )
  write.csv(all_results[[as.character(ratio)]],
            sprintf("N%d_d%d_rep%d_ratio%.2f.csv", N, d, max_iter, ratio),
            row.names = FALSE)
  cat(sprintf(">>> ratio %.2f done at %s\n", ratio, format(Sys.time())))
}

stopCluster(cluster)

# ── Summary ────────────────────────────────────────────────────────────────────
# save(all_results, file = "result_N30k_d20_rep1000.Rdata")
# load("result_N30k_d20_rep1000.Rdata")

# write.csv(all_results, "all_results_N30k_d20_rep1000.csv")
library(tidyr)
library(dplyr)
library(latex2exp)


df_all <- bind_rows(all_results)
if (length(ratios) > 1) {
  write.csv(df_all, sprintf("N%d_d%d_rep%d.csv", N, d, max_iter), row.names = FALSE)
}

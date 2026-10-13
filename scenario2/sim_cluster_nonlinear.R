# Nonlinear-baseline scenario: same as sim_cluster.R except
#   (1) xi*(X) gains 0.25 * X6 * X7 + 0.25 * (X8^2 - 1), so the GLM baseline is misspecified;
#   (2) only the GLM baseline is fitted.
library(glmnet)
library(doParallel)
library(foreach)
library(dplyr)

# ── Cluster setup ───────────────────────────────────────────────────────────
ncores  <- as.integer(Sys.getenv("NCORES", Sys.getenv("SLURM_CPUS_PER_TASK", "2")))
cluster <- makeCluster(ncores)
registerDoParallel(cluster)
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
  
  l <- log(0.1 / 0.9) - 3 * rowMeans(x[, 1:10]) +
    0.25 * x[, 6] * x[, 7] + 0.25 * (x[, 8]^2 - 1) + theta * (x[, 1] + x[, 2])
  p <- link_inv(l)
  y <- rbinom(n, 1, prob = p)
  list(x = x, y = y)
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
                 .packages = c("glmnet")) %dopar% {
                   
                   # set.seed(1000 + ratio * 10 + iter * 100)
                   set.seed(iter * 1000)
                   
                   data_1 <- get_data(ns, theta = 0, d = 20);     x_1 <- data_1$x; y_1 <- data_1$y
                   data_0 <- get_data(nt, theta = theta, d = 20); x_0 <- data_0$x; y_0 <- data_0$y
                   
                   S <- c(rep(0, nt), rep(1, ns))
                   x <- rbind(x_0, x_1)
                   y <- c(y_0, y_1)
                   
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
                   
                   # ── Coverage ────────────────────────────────────────────────────
                   cov_glm       <- as.numeric(abs(theta_glm_no - theta) <= 1.96 * sd_glm_no)
                   cov_glm_unpen <- as.numeric(abs(theta_glm_unpen - theta) <= 1.96 * sd_glm_unpen)

                   c(theta_glm_la, theta_glm_unpen, theta_glm_no,
                     sd_glm_no, sd_glm_unpen,
                     cov_glm, cov_glm_unpen)
                 }

  mat <- matrix(unlist(res), 7, max_iter)
  # row 1: theta_glm_la
  # row 2: theta_glm_unpen
  # row 3: theta_glm_no
  # row 4: sd_glm_no
  # row 5: sd_glm_unpen
  # row 6: cov_glm_no
  # row 7: cov_glm_unpen

  all_results[[as.character(ratio)]] <- data.frame(
    ratio     = ratio,
    nt        = nt,
    method    = rep(c("LinearAdj GLM", "LinearAdjUnpen GLM", "Debiased GLM"), each = max_iter),
    theta_hat = c(mat[1,], mat[2,], mat[3,]),
    sd_est    = c(rep(NA, max_iter), mat[5,], mat[4,]),
    covered   = c(rep(NA, max_iter), mat[7,], mat[6,])
  )
  write.csv(all_results[[as.character(ratio)]],
            sprintf("nonlinear_N%d_d%d_rep%d_ratio%.2f.csv", N, d, max_iter, ratio),
            row.names = FALSE)
  cat(sprintf(">>> ratio %.2f done at %s\n", ratio, format(Sys.time())))
}

stopCluster(cluster)

if (length(ratios) > 1) {
  df_all <- bind_rows(all_results)
  write.csv(df_all, sprintf("nonlinear_N%d_d%d_rep%d.csv", N, d, max_iter),
            row.names = FALSE)
}

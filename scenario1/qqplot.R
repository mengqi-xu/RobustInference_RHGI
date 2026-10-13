library(dplyr)
theta    <- 0.5
N        <- 30000
max_iter <- 1000

df_all = read.csv("all_results_N30k_d20_rep1000.csv")
df_summary <- df_all %>%
  group_by(ratio, nt, method) %>%
  summarise(
    bias     = mean(theta_hat) - theta,
    bias2    = (mean(theta_hat) - theta)^2,
    variance = var(theta_hat),
    rmse     = sqrt(mean((theta_hat - theta)^2)),
    coverage = mean(covered, na.rm = TRUE),
    .groups  = "drop"
  )

pdf("qqplot_linear.pdf", width = 13, height = 12.8)

layout_mat <- rbind(
  c(0, 10, 11, 12),
  c(13,  1,  2,  3),
  c(14,  4,  5,  6),
  c(15,  7,  8,  9)
)
layout(layout_mat,
       widths  = c(1, 4, 4, 4),
       heights = c(0.8, 4, 4, 4))

par(cex.axis = 1.2, cex.lab = 1.2)

ratios  <- c(0.4, 0.2, 0.1)
methods <- c("Debiased GLM", "Debiased XGB", "Debiased NN")

for (r in ratios) {
  for (m in methods) {
    par(mar = c(2, 2, 1, 1), pty = "s")
    vals <- (df_all[df_all$ratio == r & df_all$method == m, "theta_hat"] - 0.5) /
      sqrt(df_summary$variance[df_summary$ratio == r & df_summary$method == m])
    qqnorm(vals, main = "", xlab = "", ylab = "")
    abline(0, 1, col = "red", lwd = 1.5)
  }
}

col_labels <- c("GLM", "XGBoost", "NeuralNetwork")
for (lab in col_labels) {
  par(mar = c(0, 0, 0, 0))
  plot.new()
  text(0.5, 0.3, lab, cex = 1.8, font = 2, xpd = NA)
}

row_labels <- c("Proportion of\nMinority = 0.4",
                "Proportion of\nMinority = 0.2",
                "Proportion of\nMinority = 0.1")
for (lab in row_labels) {
  par(mar = c(0, 0, 0, 0))
  plot.new()
  text(0.5, 0.5, lab, cex = 1.5, font = 2, srt = 90, xpd = NA)
}

dev.off()
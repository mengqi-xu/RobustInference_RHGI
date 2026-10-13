# QQ plots of the standardized debiased estimator for the nonlinear-baseline scenario (Scenario 2). 
#
# Usage: Rscript qqplot_nonlinear.R [csv] [out_pdf] [standardize]
#   csv          default: nonlinear_N30000_d20_rep1000.csv
#   out_pdf      default: qqplot_nonlinear.pdf
#   standardize  "truth"     : (theta_hat - theta) / sd_est   (default)
#                "empirical" : (theta_hat - mean) / sd         (shape only)

args        <- commandArgs(trailingOnly = TRUE)
csv_file    <- if (length(args) >= 1) args[1] else "nonlinear_N30000_d20_rep1000.csv"
out_file    <- if (length(args) >= 2) args[2] else "qqplot_nonlinear.pdf"
standardize <- if (length(args) >= 3) args[3] else "truth"
stopifnot(standardize %in% c("truth", "empirical"))

theta  <- 0.5
ratios <- c(0.4, 0.2, 0.1)   

df <- read.csv(csv_file)
df <- df[df$method == "Debiased GLM", ]

pdf(out_file, width = 9, height = 3.4)
par(mfrow = c(1, length(ratios)), mar = c(3, 3, 3, 1), oma = c(0, 1.5, 0, 0),
    mgp = c(1.8, 0.6, 0), las = 0)

for (r in ratios) {
  sub <- df[abs(df$ratio - r) < 1e-8, ]
  z <- if (standardize == "truth") {
    (sub$theta_hat - theta) / sub$sd_est
  } else {
    (sub$theta_hat - mean(sub$theta_hat)) / sd(sub$theta_hat)
  }
  lim <- range(c(z, qnorm(ppoints(length(z)))))
  qqnorm(z, main = sprintf("Proportion of Minority = %g", r),
         xlab = "", ylab = "", xlim = lim, ylim = lim,
         cex = 0.7, cex.main = 1.1, font.main = 2)
  abline(0, 1, col = "red", lwd = 1.2)
}

invisible(dev.off())
cat(sprintf("Saved %s (standardize = %s)\n", out_file, standardize))

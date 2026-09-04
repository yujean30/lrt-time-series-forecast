source("scripts/00_setup.R")

y  <- load_series()
sp <- split_series(y)
train <- sp$train; test <- sp$test

candidates <- list(
  list(label = "ETS(A,N,N)",  model = "ANN", damped = NULL),
  list(label = "ETS(A,A,N)",  model = "AAN", damped = FALSE),
  list(label = "ETS(A,Ad,N)", model = "AAN", damped = TRUE),
  list(label = "ETS(A,N,A)",  model = "ANA", damped = NULL),
  list(label = "ETS(A,A,A)",  model = "AAA", damped = FALSE),
  list(label = "ETS(A,Ad,A)", model = "AAA", damped = TRUE),
  list(label = "ETS(M,N,M)",  model = "MNM", damped = NULL),
  list(label = "ETS(M,A,M)",  model = "MAM", damped = FALSE),
  list(label = "ETS(M,Ad,M)", model = "MAM", damped = TRUE)
)

results <- data.frame()
fits <- list()
forecasts <- list()

for (cand in candidates) {
  fit <- tryCatch(
    ets(train, model = cand$model, damped = cand$damped),  
    error = function(e) NULL
  )
  
  if (!is.null(fit)) {
    fc  <- forecast(fit, h = H)
    acc <- accuracy(fc, test)
    
    # Extract evaluation metrics
    train_rmse <- acc["Training set", "RMSE"]
    test_rmse  <- acc["Test set", "RMSE"]
    test_mae   <- acc["Test set", "MAE"]
    test_mape  <- acc["Test set", "MAPE"]
    test_mase  <- acc["Test set", "MASE"]
    
    # Calculate Overfitting Ratio
    rmse_ratio <- test_rmse / train_rmse
    
    # Single Ljung-Box evaluation for criteria check
    lb_test   <- Box.test(residuals(fit), lag = LAG_MAX, type = "Ljung-Box")
    lb_pvalue <- lb_test$p.value
    
    # Constraint verification
    meets_criteria <- (rmse_ratio < 1.3) && 
      (test_mape < 10) && 
      (test_mase < 1) && 
      (!is.na(lb_pvalue) && lb_pvalue > 0.05)
    
    fits[[cand$label]]      <- fit
    forecasts[[cand$label]] <- fc
    
    results <- rbind(results, data.frame(
      spec        = cand$label,
      AICc        = fit$aicc,
      BIC         = fit$bic,
      RMSE_Ratio  = round(rmse_ratio, 3),
      RMSE        = round(test_rmse, 2),
      MAE         = round(test_mae, 2),
      MAPE        = round(test_mape, 2),
      MASE        = round(test_mase, 3),
      LB_pvalue   = round(lb_pvalue, 4),
      Valid       = meets_criteria
    ))
  }
}

cat("\n--- All ETS Candidate Models Evaluated ---\n")
print(results, row.names = FALSE)

valid_results <- results[results$Valid == TRUE, ]
valid_results <- valid_results[order(valid_results$AICc), ]

if (nrow(valid_results) > 0) {
  best_label <- valid_results$spec[1]
  cat("\nSelected Best Valid Model:", best_label, " (AICc =", round(valid_results$AICc[1], 2), ")\n")
} else {
  warning("No ETS model satisfied ALL constraints (Ratio < 1.3, MAPE < 10%, MASE < 1, LB p > 0.05). Selecting lowest AICc model.")
  results <- results[order(results$AICc), ]
  best_label <- results$spec[1]
}

fit_ets <- fits[[best_label]]
print(summary(fit_ets))

fc_ets <- forecast(fit_ets, h = H)
p_fc <- autoplot(fc_ets) + autolayer(test, series = "Actual") +
  ggtitle(paste(best_label, "forecast"))
print(p_fc)
ggsave(fig("ets_forecast.png"), p_fc, width = 8, height = 5, dpi = 150)

cat("\n--- Holdout accuracy (12-month full seasonal cycle) ---\n")
print(accuracy(fc_ets, test))

cat("\n--- Residual diagnostics ---\n")
checkresiduals(fit_ets)
dev.copy(png, filename = fig("ets_residuals.png"), width = 900, height = 700, res = 130)
dev.off()
print(Box.test(residuals(fit_ets), lag = LAG_MAX, type = "Ljung-Box"))
cat("ACF-out-of-bounds:", acf_out_of_bounds(residuals(fit_ets)), "/", LAG_MAX, "\n")

g <- gap_check(accuracy(fc_ets, test))
cat("Train/test MAPE gap:", round(g$mape_gap * 100, 1),
    "% | within 10%:", g$within_10pct, "\n")

summ <- model_summary(best_label, fit_ets, fc_ets, test)
print(summ)
write.csv(results, tbl("ets_grid.csv"), row.names = FALSE)
write.csv(summ, tbl("summary_ets.csv"), row.names = FALSE)
saveRDS(fit_ets, mdl("fit_ets.rds"))
saveRDS(fc_ets,  mdl("fc_ets.rds"))

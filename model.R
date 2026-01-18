# train model.R
library(mlr3verse)
library(mlr3)
library(mlr3pipelines)
library(mlr3tuningspaces)
library(mlr3extralearners)
library(data.table)
library(readxl)
library(catboost)
library(rsample)
library(progressr)
library(jsonlite)
library(ggprism)
library(boot)
rm(list = ls())
gc()
#read data
Data <- read_excel("data_model.xlsx")
#Convert factor variable
Data$Muscle <- factor(Data$Muscle,
                      levels = c("fine", "low"),
                      labels = c("fine", "low"))
Data$Gender <- factor(Data$Gender,
                      levels = c("1", "2"),
                      labels = c("1", "2"))
#Data splitting
set.seed(123)
Data_split <- initial_split(Data, prop = 0.7, strata = Muscle)
train_data <- training(Data_split)
test_data <- testing(Data_split)
#Record factor levels
target_levels <- levels(train_data$Muscle)
#Determine the feature columns
feature_cols <- setdiff(names(train_data), "Muscle")
#Create a training task
train_task <- TaskClassif$new(
  id = "muscle_classification_train",
  backend = train_data,
  target = "Muscle",
  positive = "low"
)
#Create a learner (with preprocessing pipeline)
scale_pre <- po("scale")
CatBoost_model <- as_learner(scale_pre %>>% lrn("classif.catboost", predict_type = "prob"))
CatBoost_model$id <- "CatBoost"
# Hyperparameter Tuning
handlers(global = TRUE)  # Enable progress bar
resampling <- rsmp("cv", folds = 10)
at_catboost <- auto_tuner(
  tuner = tnr("random_search", batch_size = 10),
  learner = CatBoost_model,
  resampling = resampling,
  measure = msr("classif.auc"),
  search_space = ps(
    classif.catboost.iterations = p_int(100, 400),
    classif.catboost.learning_rate = p_dbl(0.001, 0.02),
    classif.catboost.depth = p_int(6, 10),
    classif.catboost.l2_leaf_reg = p_dbl(10, 30),
    classif.catboost.random_strength = p_dbl(1, 3),
    classif.catboost.bagging_temperature = p_dbl(0.8, 2)
  ),
  terminator = trm("evals", n_evals = 100)
)
set.seed(123)
at_res <- at_catboost$train(train_task)

#Check the optimal parameters
optimal_params <- at_res$learner$param_set$values
cat("\nOptimal parameters:\n")
print(optimal_params)
#Train the final model using optimal parameters
library(mlr3misc)
final_learner <- CatBoost_model$clone()
final_learner$param_set$values <- insert_named(final_learner$param_set$values, optimal_params)
final_learner$train(train_task)
#Calculate the optimal threshold (Youden's index)
library(pROC)
train_pred_prob <- final_learner$predict(train_task)
train_probs <- train_pred_prob$prob[, "low"]
train_labels <- train_data$Muscle
roc_obj_train <- roc(response = train_labels, predictor = train_probs)
catboost_tr_auc_tr <- auc(roc_obj_train)
#Calculate the confidence interval of the AUC for the training set
CI_catboost_tr <- ci.auc(roc_obj_train)
#Plot the ROC curve of the training set
roc_plot_catboost_tr <- ggroc(roc_obj_train, color ="#CC79A7", linewidth = 1,
                              legacy.axes=TRUE) + 
  annotate("segment", x = 0, xend = 1, y = 0, yend = 1, 
           color = "grey", linetype = "dashed") +  
  labs(
    title = paste0("CatBoost ROC Curve (AUC = ", round(catboost_tr_auc_tr, 4), ")"),
    subtitle = paste0("95% CI: ", round(CI_catboost_tr[1], 4), " - ", round(CI_catboost_tr[3], 4)),
    x = "False Positive Rate",
    y = "True Positive Rate"
  ) +
  theme_prism(border = TRUE) +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold"),
    plot.subtitle = element_text(hjust = 0.5),
    panel.grid.minor = element_blank(),
    legend.position = "none",
    axis.title.y = element_text(
      margin = margin(r = 15),  
      angle = 90,  
      vjust = 2   
    )
  ) + 
  coord_equal()
print(roc_plot_catboost_tr)
train_best_threshold_info <- coords(roc_obj_train, "best", best.method = "youden",
                                    ret = c("threshold", "sensitivity", "specificity"))
train_best_threshold <- train_best_threshold_info$threshold[1]
cat("Training Set Optimal Threshold", round(train_best_threshold, 4), "\n")
# Bootstrap confidence interval
boot_youden <- function(data, indices) {
  boot_data <- data[indices, ]
  boot_roc <- roc(boot_data$true_label, boot_data$pred_prob, quiet = TRUE)
  boot_threshold <- coords(boot_roc, "best", best.method = "youden",
                           ret = "threshold", transpose = FALSE)$threshold[1]
  return(boot_threshold)
}
data_for_boot <- data.frame(
  true_label = train_labels,
  pred_prob = train_probs
)
set.seed(123)
boot_results <- boot(data = data_for_boot, statistic = boot_youden, R = 1000)
boot_ci <- boot.ci(boot_results, conf = 0.95, type = "perc", index = 1)
cat("95% Confidence Interval Threshold:", boot_ci$percent[4], "-", boot_ci$percent[5], "\n")
#Save the model and related information
dir.create("models", showWarnings = FALSE)
#Save the model
saveRDS(final_learner, file = "models/catboost_muscle_model.rds")
#Save feature column names
saveRDS(feature_cols, file = "models/feature_cols.rds")
#Save the target factor level
saveRDS(target_levels, file = "models/target_levels.rds")
#Save threshold information
threshold_asset <- list(
  method = "youden",
  youden_threshold = as.numeric(train_best_threshold),
  boot_ci_95 = c(boot_ci$percent[4], boot_ci$percent[5]),
  boot_R = 1000,
  seed = 123
)
saveRDS(threshold_asset, "models/threshold_asset.rds")

#Save the optimal parameters
jsonlite::write_json(optimal_params, "models/best_params.json")
#Save the training set and test set (for subsequent analysis)
saveRDS(train_data, "models/train_data.rds")
saveRDS(train_data, "models/background_data.rds")
saveRDS(test_data, "models/test_data.rds")

train_pred <- final_learner$predict(train_task)
train_acc <- train_pred$score(msr("classif.acc"))
train_auc <- train_pred$score(msr("classif.auc"))
train_precision <- train_pred$score(msr("classif.precision"))
train_recall <- train_pred$score(msr("classif.recall"))
train_f1 <- train_pred$score(msr("classif.fbeta"))
train_specificity <- train_pred$score(msr("classif.specificity"))

cat("=== Training Set Performance ===\n")
cat("AUC:", round(train_auc, 4), "\n")
cat("Accuracy:", round(train_acc, 4), "\n")
cat("Precision:", round(train_precision, 4), "\n")
cat("Recall/Sensitivity:", round(train_recall, 4), "\n")
cat("Specificity:", round(train_specificity, 4), "\n")
cat("F1 Score:", round(train_f1, 4), "\n\n")

test_task <- TaskClassif$new(
  id = "test",
  backend = test_data,
  target = "Muscle",
  positive = "low"
)

test_pred_prob <- final_learner$predict(test_task)
test_pred <- final_learner$predict(test_task)
test_pred_trainthr <- test_pred_prob$set_threshold(train_best_threshold)
test_auc <- test_pred_prob$score(msr("classif.auc"))
test_acc <- test_pred_trainthr$score(msr("classif.acc"))
test_precision <- test_pred_trainthr$score(msr("classif.precision"))
test_recall <- test_pred_trainthr$score(msr("classif.recall"))       # = sensitivity
test_f1 <- test_pred_trainthr$score(msr("classif.fbeta"))
test_specificity <- test_pred_trainthr$score(msr("classif.specificity"))
test_sensitivity <- test_recall

cat("=== Test set performance (using the optimal threshold from the training set) ===\n")
cat("F1 Score:",       round(test_f1, 4), "\n")
cat("Accuracy:",       round(test_acc, 4), "\n")
cat("Recall / Sensitivity:", round(test_recall, 4), "\n")
cat("Precision:",      round(test_precision, 4), "\n")
cat("Specificity:",    round(test_specificity, 4), "\n")
cat("AUC:",            round(test_auc, 4), "\n")

catboost_probs <- test_pred$prob[, "low"]
catboost_labels <- test_data$Muscle
roc_obj_catboost <- roc(response = catboost_labels, predictor = catboost_probs)
catboost_auc <- auc(roc_obj_catboost)
CI_catboost <- ci.auc(roc_obj_catboost)
#Draw the ROC curve of the test set
roc_plot_catboost <- ggroc(roc_obj_catboost, color ="#CC79A7", linewidth = 1,legacy.axes=TRUE) + 
  annotate("segment", x = 0, xend = 1, y = 0, yend = 1, 
           color = "grey", linetype = "dashed") +  
  labs(
    title = paste0("CatBoost ROC Curve (AUC = ", round(catboost_auc, 4), ")"),
    subtitle = paste0("95% CI: ", round(CI_catboost[1], 4), " - ", round(CI_catboost[3], 4)),
    x = "False Positive Rate",
    y = "True Positive Rate"
  ) +
  theme_prism(border = TRUE) +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold"),
    plot.subtitle = element_text(hjust = 0.5),
    panel.grid.minor = element_blank(),
    legend.position = "none",
    axis.title.y = element_text(
      margin = margin(r = 15), 
      angle = 90, 
      vjust = 2  
    )
  ) + 
  coord_equal()
print(roc_plot_catboost)

#External validation
External_data <- read_excel("validation set.xlsx")
External_data$Muscle <- factor(External_data$Muscle,
                                 levels = c("fine","low"),
                                 labels = c("fine", "low"))
External_data$Gender <- factor(External_data$Gender,
                             levels = c("1","2"),
                             labels = c("1", "2"))
exval_task <- TaskClassif$new(
  id = "exval",
  backend = External_data,
  target = "Muscle",
  positive = "low"
)

exval_pred_prob <- final_learner$predict(exval_task)
exval_pred_trainthr <- exval_pred_prob$set_threshold(train_best_threshold)
exval_auc <- exval_pred_prob$score(msr("classif.auc"))
exval_acc <- exval_pred_trainthr$score(msr("classif.acc"))
exval_precision <- exval_pred_trainthr$score(msr("classif.precision"))
exval_recall <- exval_pred_trainthr$score(msr("classif.recall"))    
exval_f1 <- exval_pred_trainthr$score(msr("classif.fbeta"))
exval_specificity <- exval_pred_trainthr$score(msr("classif.specificity"))
exval_sensitivity <- exval_recall 

cat("\n=== External Validation (NHANES) Performance Metrics (Using the Optimal Threshold from the Training Set) ===\n")
cat("F1 Score:", round(exval_f1, 4), "\n")
cat("Accuracy:", round(exval_acc, 4), "\n")
cat("Recall / Sensitivity:", round(exval_recall, 4), "\n")
cat("Precision:", round(exval_precision, 4), "\n")
cat("Specificity:", round(exval_specificity, 4), "\n")
cat("AUC:", round(exval_auc, 4), "\n")
cat("Sensitivity:", round(exval_sensitivity, 4), "\n")

val_catboost_probs <- exval_pred_prob$prob[, "low"]
val_catboost_labels <- External_data$Muscle
roc_obj_val_catboost <- roc(response = val_catboost_labels, 
                            predictor = val_catboost_probs)
val_catboost_auc <- auc(roc_obj_val_catboost)
CI_val_catboost <- ci.auc(roc_obj_val_catboost)
val_catboost_plot <- ggroc(roc_obj_val_catboost, color ="#CC79A7", linewidth = 1) + 
  annotate("segment", x = 1, xend = 0, y = 0, yend = 1, 
           color = "grey", linetype = "dashed") +  
  labs(
    title = paste0("CatBoost ROC Curve (AUC = ", round(val_catboost_auc, 4), ")"),
    subtitle = paste0("95% CI: ", round(CI_val_catboost[1], 4), " - ", round(CI_val_catboost[3], 4)),
    x = "False Positive Rate",
    y = "True Positive Rate"
  ) +
  theme_prism(border = TRUE) +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold"),
    plot.subtitle = element_text(hjust = 0.5),
    panel.grid.minor = element_blank(),
    legend.position = "none",
    axis.title.y = element_text(
      margin = margin(r = 15),  
      angle = 90, 
      vjust = 2  
    )
  ) + 
  coord_equal()
print(val_catboost_plot)

library(dcurves)
library(rmda)
df_dca_test <- data.frame(
  outcome = catboost_labels,            
  test_prob=catboost_probs
)
df_dca_test$outcome <- ifelse(df_dca_test$outcome=="low",1,0)
df_dca_val <- data.frame(
  outcome = val_catboost_labels,            
  val_prob=val_catboost_probs
)
df_dca_val$outcome <- ifelse(df_dca_val$outcome=="low",1,0)
dca_val <- decision_curve(outcome ~ val_prob,
                          data = df_dca_val,
                          thresholds = seq(0, 0.8, by = 0.01),
                          confidence.intervals = NA)
plot_decision_curve(dca_val,
                    curve.names = "External validation data",
                    col ="#9F7EC9")
dca_test <- decision_curve(outcome ~ test_prob, 
                           data = df_dca_test,
                           thresholds = seq(0, 0.8, by = 0.01),
                           confidence.intervals = NA)
plot_decision_curve(dca_test,
                    curve.names = "Test data",
                    col ="#BA5F9A")
library(kernelshap)
library(shapviz)
library(ggplot2)
library(DALEX)
library(shapviz)
library(DALEXtra)
bg_X <- train_data[, -which(names(train_data) == "Muscle")]
X <- test_data[, -which(names(test_data) == "Muscle")]
ps <- permshap(
  final_learner, X = test_data[,feature_cols], bg_X=bg_X, predict_type = "prob"
)
# 5. 生成 shapviz 对象
ps_result <- shapviz(ps)
ps_result[["low"]]
ps_result
p_beeswarm <- sv_importance(ps_result[["low"]] , kind = "beeswarm", show_numbers = FALSE,
                            bee_width= 0.3, 
                            alpha  = 0.7)  +     
  scale_color_gradientn(
    colors = c("#116694", "white", "#861925"),
    guide = guide_colorbar(
      direction = "vertical",      
      barheight = unit(4, "cm"), 
      barwidth = unit(0.4, "cm"),   
      label.position = "right",    
      label.theme = element_text(size = 12),
      frame.colour = "black",     
      frame.linewidth = 0.5
    ))+
  theme_prism(border = TRUE) +
  theme(
    text = element_text(face = "bold"),   
    axis.text.y = element_text(size = 12, color = "black"),
    axis.title.x = element_text(size = 12, face = "bold"),
    legend.position = "right",
    panel.grid.minor = element_blank(),
    aspect.ratio= 1,
    legend.text = element_text(face = "bold", size = 13)
  ) +
  labs(
    x = "SHAP value (impact on model output)"
  )

print(p_beeswarm)


p_imp <- sv_importance(ps_result[["low"]], kind = "bar", fill = "#116694") + 
  theme_prism(border = TRUE) +
  theme(
    axis.text.y = element_text(size = 13),
    plot.title = element_text(face = "bold"),
    aspect.ratio= 1
  ) 

print(p_imp)

plot_waterfall <- sv_waterfall(ps_result[["low"]]) +
  theme_prism(
    border = TRUE,
    base_size = 13
  ) +
  theme(
    text = element_text(face = "bold"),
    axis.text = element_text(face = "bold"),
    axis.title = element_text(face = "bold"),
    legend.text = element_text(face = "bold"),
    legend.title = element_text(face = "bold"),
    legend.position = "bottom",
    aspect.ratio = 1
  ) +
  scale_fill_brewer(palette = "Blues") 
print(plot_waterfall)

sv_dependence2D <- sv_dependence2D(ps_result[["low"]], x = "BMI", y = c("Hb","ALT","WBC"))& 
  geom_smooth(method = "loess", se = FALSE, color = "darkred", linetype = "dashed", 
              linewidth = 0.6,alpha=0.6) &
  scale_color_gradientn(
    colors = c("#116694", "white", "#861925"),
    guide = guide_colorbar(
      direction = "vertical",      
      barheight = unit(4, "cm"), 
      barwidth = unit(0.4, "cm"), 
      label.position = "right",     
      title.position = "top",
      label.theme = element_text(size = 12),
      frame.colour = "black",     
      frame.linewidth = 0.5
    )
  ) &
  theme_prism(border = TRUE)&
  theme(
    legend.title.position = "top",
    legend.position = "right", 
    legend.text = element_text(face = "bold"),
    axis.title.x = element_text(face = "bold"), 
    axis.title = element_text(face = "bold"),
    aspect.ratio= 1
  ) 
print(sv_dependence2D)
sv_dependence2D<- sv_dependence2D &
  theme(
    legend.position = "right",
    legend.key.height = unit(0.4, "cm"),
    legend.key.width = unit(0.2, "cm"),
    legend.text = element_text(size = 13),
    legend.title = element_text(size = 13),
    legend.margin = ggplot2::margin(t = 0, r = 0, b = 0, l = 0, unit = "pt"),
    panel.spacing = unit(0.05, "cm")
  )
print(sv_dependence2D)

p_dep_wbc <- sv_dependence(ps_result[["low"]], v = "WBC",color_var = "BMI",share_y = FALSE) + # 指定颜色映射变量
  geom_smooth(method = "loess", se = FALSE, color = "darkred", linetype = "dashed", 
              linewidth = 0.6,alpha=0.6) +
  scale_color_gradientn(
    colors = c("#116694", "white", "#861925"),
    guide = guide_colorbar(
      direction = "vertical",       
      barheight = unit(4, "cm"), 
      barwidth = unit(0.4, "cm"),   
      label.position = "right",    
      label.theme = element_text(size = 12),
      frame.colour = "black",     
      frame.linewidth = 0.5
    )
  ) +
  theme_prism(border = TRUE) +
  theme(
    legend.justification =0.5,
    legend.position = "right",
    axis.title = element_text(face = "bold",size = 13),
    legend.box.margin = ggplot2::margin(t = 0, r = 0, b = 0, l = 0, unit = "pt"),
    aspect.ratio= 1,
    legend.text = element_text(face = "bold", size = 13),
    legend.key.width = unit(1.2, "cm"),
    legend.key.height = unit(0.25, "cm"),
    legend.spacing.x = unit(0.1, "cm"),  
    legend.spacing.y = unit(0.1, "cm"),   
    legend.box.spacing = unit(0.2, "cm"), 
  ) +
  geom_hline(yintercept = 0, linetype = "dotted", color = "gray50") + # 添加0线
  labs(
    y = "SHAP Value for WBC"
  )
print(p_dep_wbc)

pb_wbc <- ggplot_build(p_dep_wbc)
smooth_data_wbc <- pb_wbc$data[[2]] 
head(smooth_data_wbc) 
crossings_wbc <- which(diff(sign(smooth_data_wbc$y)) != 0)
intercepts_wbc <- data.frame(x_intercept = numeric(), y_intercept = numeric())
for (i in crossings_wbc) {
  x1 <- smooth_data_wbc$x[i]
  y1 <- smooth_data_wbc$y[i]
  x2 <- smooth_data_wbc$x[i + 1]
  y2 <- smooth_data_wbc$y[i + 1]
  x_cross <- x1 + (0 - y1) * (x2 - x1) / (y2 - y1)
  intercepts_wbc <- rbind(intercepts_wbc, data.frame(x_intercept = x_cross, y_intercept = 0))
}
print("The coordinates of the intersection points of the smooth curve with y=0 are：")
print(intercepts_wbc)


p_dep_Hb <- sv_dependence(ps_result[["low"]], v = "Hb",color_var = "BMI",share_y = FALSE) + 

  geom_smooth(method = "loess", se = FALSE, color = "darkred", linetype = "dashed", 
              linewidth = 0.6,alpha=0.6) +
  scale_color_gradientn(
    colors = c("#116694", "white", "#861925"),
    guide = guide_colorbar(
      direction = "vertical",      
      barheight = unit(4, "cm"), 
      barwidth = unit(0.4, "cm"),  
      label.position = "right",    
      label.theme = element_text(size = 12),
      frame.colour = "black",     
      frame.linewidth = 0.5
    )
  ) +
  theme_prism(border = TRUE) +
  theme(
    legend.justification =0.5,
    legend.position = "right", 
    axis.title = element_text(face = "bold",size = 13),
    legend.box.margin = ggplot2::margin(t = 0, r = 0, b = 0, l = 0, unit = "pt"),
    aspect.ratio= 1,
    legend.text = element_text(face = "bold", size = 13),
    legend.key.width = unit(1.2, "cm"),
    legend.key.height = unit(0.25, "cm"),
    legend.spacing.x = unit(0.1, "cm"),   
    legend.spacing.y = unit(0.1, "cm"),  
    legend.box.spacing = unit(0.2, "cm")
  ) +
  geom_hline(yintercept = 0, linetype = "dotted", color = "gray50") + 
  labs(
    y = "SHAP Value for Hb"
  )

print(p_dep_Hb)

pb_Hb <- ggplot_build(p_dep_Hb)
smooth_data_Hb <- pb_Hb$data[[2]] 
head(smooth_data_Hb) 
crossings_Hb <- which(diff(sign(smooth_data_Hb$y)) != 0)
intercepts_Hb <- data.frame(x_intercept = numeric(), y_intercept = numeric())
for (i in crossings_Hb) {
  x1 <- smooth_data_Hb$x[i]
  y1 <- smooth_data_Hb$y[i]
  x2 <- smooth_data_Hb$x[i + 1]
  y2 <- smooth_data_Hb$y[i + 1]
  x_cross <- x1 + (0 - y1) * (x2 - x1) / (y2 - y1)
  intercepts_Hb <- rbind(intercepts_Hb, data.frame(x_intercept = x_cross, y_intercept = 0))
}

print("The coordinates of the intersection points of the smooth curve with y=0 are：")
print(intercepts_Hb)

p_dep_ALT <- sv_dependence(ps_result[["low"]], v = "ALT",color_var = "BMI",share_y = FALSE) + 
  geom_smooth(method = "loess", se = FALSE, color = "darkred", linetype = "dashed", 
              linewidth = 0.6,alpha=0.6) +
  scale_color_gradientn(
    colors = c("#116694", "white", "#861925"),
    guide = guide_colorbar(
      direction = "vertical",       
      barheight = unit(4, "cm"), 
      barwidth = unit(0.4, "cm"),  
      label.position = "right",    
      label.theme = element_text(size = 12),
      frame.colour = "black",   
      frame.linewidth = 0.5
    )
  ) +
  theme_prism(border = TRUE) +
  theme(
    legend.justification =0.5,
    legend.position = "right", 
    axis.title = element_text(face = "bold",size = 13),
    legend.box.margin = ggplot2::margin(t = 0, r = 0, b = 0, l = 0, unit = "pt"),
    aspect.ratio= 1,
    legend.text = element_text(face = "bold", size = 13),
    legend.key.width = unit(1.2, "cm"),
    legend.key.height = unit(0.25, "cm"),
    legend.spacing.x = unit(0.1, "cm"),   
    legend.spacing.y = unit(0.1, "cm"),   
    legend.box.spacing = unit(0.2, "cm")
  ) +
  geom_hline(yintercept = 0, linetype = "dotted", color = "gray50") +
  labs(
    y = "SHAP Value for ALT"
  )

print(p_dep_ALT)

pb_ALT <- ggplot_build(p_dep_ALT)
smooth_data_ALT <- pb_ALT$data[[2]] 
head(smooth_data_ALT) 
crossings_ALT <- which(diff(sign(smooth_data_ALT$y)) != 0)
intercepts_ALT <- data.frame(x_intercept = numeric(), y_intercept = numeric())
for (i in crossings_ALT) {
  x1 <- smooth_data_ALT$x[i]
  y1 <- smooth_data_ALT$y[i]
  x2 <- smooth_data_ALT$x[i + 1]
  y2 <- smooth_data_ALT$y[i + 1]
  x_cross <- x1 + (0 - y1) * (x2 - x1) / (y2 - y1)
  intercepts_ALT <- rbind(intercepts_ALT, data.frame(x_intercept = x_cross, y_intercept = 0))
}

print("The coordinates of the intersection points of the smooth curve with y=0 are：")
print(intercepts_ALT)

p_dep <- (p_dep_Hb|p_dep_ALT|p_dep_wbc)&
  theme(
    legend.position = "right",
    legend.key.height = unit(0.4, "cm"),
    legend.key.width = unit(0.2, "cm"),
    legend.text = element_text(size = 13),
    legend.title = element_text(size = 13),
    legend.margin = ggplot2::margin(t = 0, r = 0, b = 0, l = 0, unit = "pt"),
    panel.spacing = unit(0.05, "cm")
  )
print(p_dep)

library(mlr3verse)
library(mlr3)
library(data.table)
library(readxl)
library(tidyr)
library(writexl)
library(mlr3pipelines)
library(future)
library(mlr3benchmark)
library(tidyplots)
library(mlr3tuningspaces)
library(ggplot2)
library(ggprism)
library(ggtext)
library(mlr3extralearners)
library(rsample)
library(dplyr)
library(mlr3filters)
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
feature_cols <- setdiff(names(train_data), "Muscle")
plan("multisession", workers = 6)  # Adjust according to your computer configuration
# 创建训练任务
#Create a training task
train_task <- TaskClassif$new(
  id = "muscle_classification_train",
  backend = train_data,
  target = "Muscle",
  positive = "low"
)
scale_pre <- po("scale")
#nnet
nnet_model <- as_learner(scale_pre %>>% lrn("classif.nnet", 
                                            predict_type = "prob"))
nnet_model$id <- "nnet"
at_nnet <- auto_tuner(
  tuner = tnr("random_search", 
              batch_size = 10),
  learner = nnet_model,
  resampling = rsmp("cv", folds = 10),
  measure = msr("classif.auc"),
  search_space = ps(
    classif.nnet.size = p_int(lower = 1, upper = 8),
    classif.nnet.maxit = p_int(lower = 30, upper = 80),
    classif.nnet.decay = p_dbl(lower = 1e-2, upper = 1.0, logscale = TRUE)
  ),
  terminator = trm("evals", n_evals = 100)
)

#catboost
CatBoost_model <- as_learner(scale_pre %>>% lrn("classif.catboost", predict_type="prob")) 
CatBoost_model$id <- "CatBoost"
at_catboost <- auto_tuner(
  tuner = tnr("random_search", 
              batch_size = 10),
  learner = CatBoost_model,
  resampling = rsmp("cv", folds = 10),
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

#Logistic Regression
log_model <-as_learner(scale_pre %>>% lrn("classif.log_reg", predict_type="prob")) 
log_model$id <- "Logistic"
at_log <- auto_tuner(
  tuner = tnr("random_search", 
              batch_size = 10),
  learner = log_model,
  resampling = rsmp("cv", folds = 10),
  measure = msr("classif.auc"),
  search_space = ps(
    classif.log_reg.epsilon = p_dbl(lower = 1e-8, upper = 1e-4, logscale = TRUE),
    classif.log_reg.maxit = p_int(lower = 10, upper = 1000)
  ),
  terminator = trm("evals", n_evals = 100)
)

#Decision Tree
tree_model <- as_learner(scale_pre %>>% lrn("classif.rpart", predict_type="prob")) 
tree_model$id <- "DecisionTree"
at_Dtree <- auto_tuner(
  tuner = tnr("random_search", 
              batch_size = 10),
  learner = tree_model,
  resampling = rsmp("cv", folds = 10),
  measure = msr("classif.auc"),
  search_space=ps(
    classif.rpart.cp = p_dbl(0.001, 0.1, logscale = TRUE),    
    classif.rpart.maxdepth = p_int(1, 30),                      
    classif.rpart.minsplit = p_int(2, 50),                     
    classif.rpart.minbucket = p_int(1, 20)                   
  ),
  terminator = trm("evals", n_evals = 100) 
)
#KNN
kknn_model <- as_learner(scale_pre %>>% lrn("classif.kknn", predict_type="prob")) 
kknn_model$id <- "KNN"
at_KNN <- auto_tuner(
  tuner = tnr("random_search", 
              batch_size = 10),
  learner = kknn_model,
  resampling = rsmp("cv", folds = 10),
  measure = msr("classif.auc"),
  search_space = ps(
    classif.kknn.k = p_int(1, 30),
    classif.kknn.distance=p_dbl(1, 5)
  ),
  terminator = trm("evals", n_evals = 100)  # 最多评估100组参数
)

#naive_bayes
nb_model <- as_learner(scale_pre %>>% lrn("classif.naive_bayes", predict_type = "prob"))
nb_model$id <- "Naive Bayes"
at_nb <- auto_tuner(
  tuner = tnr("random_search", 
              batch_size = 10),
  learner = nb_model,
  resampling = rsmp("cv", folds = 10),
  measure = msr("classif.auc"),
  search_space = ps(
    classif.naive_bayes.laplace = p_dbl(0.1, 5), 
    classif.naive_bayes.eps = p_dbl(1e-9, 0.1),       
    classif.naive_bayes.threshold = p_dbl(0.005, 0.1)  
  ),
  terminator = trm("evals", n_evals = 100)  
)

#lightGBM
lgb_model <- as_learner(scale_pre %>>% lrn("classif.lightgbm", 
                                           predict_type = "prob"))
lgb_model$id <- "LightGBM"
at_lgb <- auto_tuner(
  tuner = tnr("random_search", 
              batch_size = 10),
  learner = lgb_model,
  resampling = rsmp("cv", folds = 10),
  measure = msr("classif.auc"),
  search_space = ps(
    classif.lightgbm.num_iterations = p_int(200, 1200),
    classif.lightgbm.learning_rate = p_dbl(0.005, 0.2, logscale = TRUE),
    classif.lightgbm.num_leaves = p_int(31, 255),
    classif.lightgbm.max_depth = p_int(5, 15),
    classif.lightgbm.min_data_in_leaf = p_int(10, 100),
    classif.lightgbm.lambda_l1 = p_dbl(1e-8, 10.0, logscale = TRUE),
    classif.lightgbm.lambda_l2 = p_dbl(1e-8, 10.0, logscale = TRUE),
    classif.lightgbm.min_gain_to_split = p_dbl(1e-8, 1.0, logscale = TRUE)
  ),
  terminator = trm("evals", n_evals = 100)
)
#Enable progress bar
library(progressr)
handlers(global = TRUE)
lgr::get_logger("mlr3")$set_threshold("warn")
lgr::get_logger("bbotk")$set_threshold("warn")
set.seed(123)
design <- benchmark_grid(
  tasks = train_task,
  learners = list(at_nnet,at_catboost,at_log,at_Dtree,
                  at_KNN,at_nb,at_lgb),
  resampling = rsmp("cv", folds = 10)
)

set.seed(123)
bmr <- benchmark(design,store_models = T)
bmr
bmr$aggregate()
bmr$score(list(msr("classif.auc")))
bmr_score <- bmr$score(msrs(c(
  "classif.auc",      
  "classif.acc",     
  "classif.bbrier",   
  "classif.precision",
  "classif.recall",   
  "classif.fbeta"
)))
scores_df <- as.data.frame(bmr_score)
plan(sequential)
write_xlsx(scores_df, "bmr_scores.xlsx")
bma = bmr$aggregate(msr("classif.auc"))
bma
library(mlr3inferr)
# alpha = 0.05 is also the default
msr_ci = msr("ci.wald_cv", msr("classif.acc"), alpha = 0.05)
bmr$aggregate(msr_ci)
bmrDt <- as.data.table(bmr)
bmrModels <- mlr3misc::map(bmrDt$learner, "model")
outerLearners <- mlr3misc::map(bmrDt$learner, "learner")
length(outerLearners)

#Visualization
library(ggplot2)
library(ggprism)
library(plyr)
library(dplyr)
library(tidyplots)
auc <- bmr$score(list(msr("classif.auc"))) %>% 
  as.data.table() %>%
  select(learner_id, classif.auc)
plot_compare_auc <- auc|>
  tidyplot(x = learner_id, y = classif.auc, color = learner_id) |>  
  adjust_colors(colors_discrete_friendly_long)|>
  add_boxplot()+
  labs(x="Model", y="classif.auc")+ 
  theme_prism(border = TRUE) +
  theme(legend.position = "right",
        text = element_text(face = "bold",size = 13),
        axis.title = element_text(face = "bold",size = 13),
        axis.text.x = element_text(angle = 90,hjust = 1), 
        axis.title.x = element_blank())
print(plot_compare_auc)

acc <- bmr$score(list(msr("classif.acc"))) %>% 
  as.data.table() %>%
  select(learner_id, classif.acc)
plot_compare_acc <- acc|>
  tidyplot(x = learner_id, y = classif.acc, color = learner_id) |>  
  adjust_colors(colors_discrete_friendly_long)|>
  add_boxplot()+
  labs(x="Model", y="classif.acc")+ 
  theme_prism(border = TRUE) +
  theme(legend.position = "right",
        text = element_text(face = "bold",size = 13),
        axis.title = element_text(face = "bold",size = 13),
        axis.text.x = element_text(angle = 90,hjust = 1), 
        axis.title.x = element_blank())
print(plot_compare_acc)

bbrier <- bmr$score(list(msr("classif.bbrier"))) %>% 
  as.data.table() %>%
  select(learner_id, classif.bbrier)
plot_compare_bbrier <- bbrier|>
  tidyplot(x = learner_id, y =classif.bbrier, color = learner_id) |>  
  adjust_colors(colors_discrete_friendly_long)|>
  add_boxplot()+
  labs(x="Model", y="classif.bbrier")+  
  theme_prism(border = TRUE) +
  theme(legend.position = "right",
        text = element_text(face = "bold",size = 13),
        axis.title = element_text(face = "bold",size = 13),
        axis.text.x = element_text(angle = 90,hjust = 1), 
        axis.title.x = element_blank())
print(plot_compare_bbrier)

precision <- bmr$score(list(msr("classif.precision"))) %>% 
  as.data.table() %>%
  select(learner_id, classif.precision)
plot_compare_precision <- precision|>
  tidyplot(x = learner_id, y =classif.precision, color = learner_id) |>  
  adjust_colors(colors_discrete_friendly_long)|>
  add_boxplot()+
  labs(x="Model", y="classif.precision")+
  theme_prism(border = TRUE) +
  theme(legend.position = "right",
        text = element_text(face = "bold",size = 13),
        axis.title = element_text(face = "bold",size = 13),
        axis.text.x = element_text(angle = 90,hjust = 1), 
        axis.title.x = element_blank())
print(plot_compare_precision)

fbeta <- bmr$score(list(msr("classif.fbeta"))) %>% 
  as.data.table() %>%
  select(learner_id, classif.fbeta)
plot_compare_fbeta <- fbeta|>
  tidyplot(x = learner_id, y =classif.fbeta, color = learner_id) |>  
  adjust_colors(colors_discrete_friendly_long)|>
  add_boxplot()+
  labs(x="Model", y="classif.fbeta")+  
  theme_prism(border = TRUE) +
  theme(legend.position = "right",
        text = element_text(face = "bold",size = 13),
        axis.title = element_text(face = "bold",size = 13),
        axis.text.x = element_text(angle = 90,hjust = 1), 
        axis.title.x = element_blank())
print(plot_compare_fbeta)

# Set CRAN repository
options(repos = c(CRAN = "https://cloud.r-project.org"))

# --- Install Packages if Not Already Installed ---
packages <- c("readr", "skimr", "ggplot2", "tidyverse", "GGally",
              "rnaturalearth", "rnaturalearthdata", "ggforce", "rpart",
              "rpart.plot", "pROC", "caret", "e107","randomForest", "recipes",
              "dplyr", "keras", "tensorflow")
new.packages <- packages[!(packages %in% installed.packages()[, "Package"])]
if(length(new.packages)) {
  install.packages(new.packages)
}

# --- Load Libraries ---
library(readr)
library(skimr)
library(ggplot2)
library(tidyverse)
library(GGally)
library(e1071)
library(rnaturalearth)
library(rnaturalearthdata)
library(ggforce)
library(rpart)
library(rpart.plot)
library(pROC)
library(caret)
library(randomForest)
library(recipes)
library(dplyr)
library(keras)
library(tensorflow)


# --- Install and Configure Keras/TensorFlow if Needed ---
if (!keras::is_keras_available()) {
  install_keras()
}
if (tf_version() < "1.0") {
  install_tensorflow()
}

# --- Set Reproducible Seed ---
set.seed(123)

# --- Read and Inspect Data ---
df <- read_csv("https://www.louisaslett.com/Courses/MISCADA/bank_personal_loan.csv")
skim(df)
# Optionally, if interactive, view the data:
# View(df)

# --- Remove Irrelevant Variables and Adjust Data ---
# Remove: "ID", "ZIP.Code", "Experience", "Family", "Online"
df <- df[, !(names(df) %in% c("ID", "ZIP.Code", "Experience", "Family", "Online"))]
df$Personal.Loan <- as.factor(df$Personal.Loan)
str(df)

# --- Split Data into Training and Test Sets (70% / 30%) ---
train_ratio <- 0.7
train_indices <- sample(1:nrow(df), size = floor(train_ratio * nrow(df)), replace = FALSE)
trainset <- df[train_indices, ]
testset <- df[-train_indices, ]
cat("Training Set Rows:", nrow(trainset), "\n")
cat("Test Set Rows:", nrow(testset), "\n")

# --- Plot Histograms for All Numeric Variables ---
for (col in names(df)) {
  if (is.numeric(df[[col]])) {
    p <- ggplot(df, aes(x = .data[[col]])) +
      geom_histogram(fill = "blue", color = "black", bins = 30, alpha = 0.7) +
      labs(title = paste("Distribution of", col), x = col, y = "Count") +
      theme_minimal()
    print(p)  # Explicitly print the plot
    Sys.sleep(0.5)  # Pause briefly so you can see each plot
  }
}

# --- Plot Boxplots for All Numeric Variables (to Detect Outliers) ---
for (col in names(df)) {
  if (is.numeric(df[[col]])) {
    p <- ggplot(df, aes(x = "", y = .data[[col]])) +
      geom_boxplot(fill = "red", color = "black", alpha = 0.6) +
      labs(title = paste("Boxplot of", col), x = "", y = col) +
      theme_minimal()
    print(p)
    Sys.sleep(0.5)
  }
}

# --- Correlation Matrix of Numeric Features ---
numeric_vars <- df %>% select(where(is.numeric))
print(ggpairs(numeric_vars,
              lower = list(continuous = wrap("points", alpha = 0.4)),
              title = "Correlation Matrix of Numeric Features"))

# --- Scatter Plots: Relationship Between Important Variables and Loan Approval ---
important_vars <- c("Income", "CCAvg", "Mortgage")
# Convert Personal.Loan to numeric (0/1) for scatter plotting:
loan_numeric <- as.numeric(as.character(df$Personal.Loan))
for (var in important_vars) {
  p <- ggplot(df, aes(x = .data[[var]], y = loan_numeric)) +
    geom_point(alpha = 0.5, color = "blue") +
    geom_smooth(method = "lm", se = FALSE, color = "red") +
    labs(title = paste("Relationship between", var, "and Loan Approval"),
         x = var, y = "Loan Approval (0 or 1)") +
    theme_minimal()
  print(p)
  Sys.sleep(0.5)
}

# --- Logistic Regression Model ---
loan_fit <- glm(Personal.Loan ~ ., data = trainset, family = binomial)
coefs <- summary(loan_fit)$coefficients
coefs_df <- data.frame(
  variable = rownames(coefs),
  estimate = coefs[, 1],
  std_error = coefs[, 2],
  z_value  = coefs[, 3],
  p_value  = coefs[, 4]
)
coefs_df <- subset(coefs_df, variable != "(Intercept)")
coefs_df$variable <- reorder(coefs_df$variable, abs(coefs_df$estimate))
ggplot(coefs_df, aes(x = variable, y = estimate, fill = estimate > 0)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  labs(title = "Logistic Regression Coefficients", x = "Predictor", y = "Coefficient Estimate") +
  scale_fill_manual(values = c("red", "blue"), guide = FALSE) +
  theme_minimal()

# --- Income Brackets and 100% Stacked Bar Chart ---
df_bracketed <- df %>%
  mutate(IncomeBracket = case_when(
    Income < 50 ~ "<50K",
    Income >= 50 & Income < 100 ~ "50K-100K",
    Income >= 100 & Income < 150 ~ "100K-150K",
    TRUE ~ ">150K"
  ))
p <- ggplot(df_bracketed, aes(x = IncomeBracket, fill = Personal.Loan)) +
  geom_bar(position = "fill") +
  scale_y_continuous(labels = scales::percent_format()) +
  labs(title = "Loan Acceptance Rate by Income Bracket",
       x = "Income Bracket", y = "Percentage", fill = "Personal.Loan") +
  theme_minimal()
print(p)

# --- Age Groups and 100% Stacked Bar Chart ---
df_age <- df %>%
  mutate(AgeGroup = case_when(
    Age < 60 ~ "18-60",
    TRUE ~ "60+"
  ))
print(table(df_age$AgeGroup))
p_age <- ggplot(df_age, aes(x = AgeGroup, fill = Personal.Loan)) +
  geom_bar(position = "fill") +
  scale_y_continuous(labels = scales::percent_format()) +
  labs(title = "Loan Acceptance Rate by Age Group",
       x = "Age Group", y = "Percentage", fill = "Personal.Loan") +
  theme_minimal()
print(p_age)

# --- Evaluate Logistic Regression Model (ROC Curves) ---
train_probs <- predict(loan_fit, newdata = trainset, type = "response")
test_probs  <- predict(loan_fit, newdata = testset, type = "response")
train_roc <- roc(trainset$Personal.Loan, train_probs)
train_auc <- auc(train_roc)
cat("Training AUC:", train_auc, "\n")
plot(train_roc, main = paste("ROC Curve - Training Set (AUC =", round(train_auc, 3), ")"))
test_roc <- roc(testset$Personal.Loan, test_probs)
test_auc <- auc(test_roc)
cat("Test AUC:", test_auc, "\n")
plot(test_roc, main = paste("ROC Curve - Test Set (AUC =", round(test_auc, 3), ")"))

# --- Compare Single-Predictor and Multi-Predictor Logistic Models ---
multi_model <- glm(Personal.Loan ~ Income + CCAvg + Mortgage, data = df, family = binomial)
income_model <- glm(Personal.Loan ~ Income, data = df, family = binomial)
ccavg_model <- glm(Personal.Loan ~ CCAvg, data = df, family = binomial)
mortgage_model <- glm(Personal.Loan ~ Mortgage, data = df, family = binomial)
multi_probs <- predict(multi_model, type = "response")
income_probs <- predict(income_model, type = "response")
ccavg_probs <- predict(ccavg_model, type = "response")
mortgage_probs <- predict(mortgage_model, type = "response")
multi_roc <- roc(df$Personal.Loan, multi_probs)
income_roc <- roc(df$Personal.Loan, income_probs)
ccavg_roc <- roc(df$Personal.Loan, ccavg_probs)
mortgage_roc <- roc(df$Personal.Loan, mortgage_probs)
plot(multi_roc, col = "red", main = "ROC Curves", lwd = 2)
plot(income_roc, add = TRUE, col = "blue", lwd = 2)
plot(ccavg_roc, add = TRUE, col = "green", lwd = 2)
plot(mortgage_roc, add = TRUE, col = "orange", lwd = 2)
abline(a = 0, b = 1, lty = 2, col = "gray")
multi_auc <- round(auc(multi_roc), 3)
income_auc <- round(auc(income_roc), 3)
ccavg_auc <- round(auc(ccavg_roc), 3)
mortgage_auc <- round(auc(mortgage_roc), 3)
legend("bottomright",
       legend = c(paste("Multi-predictor (AUC =", multi_auc, ")"),
                  paste("Income only (AUC =", income_auc, ")"),
                  paste("CCAvg only (AUC =", ccavg_auc, ")"),
                  paste("Mortgage only (AUC =", mortgage_auc, ")")),
       col = c("red", "blue", "green", "orange"),
       lty = 1, lwd = 2)

# --- Confusion Matrix for Logistic Regression ---
pred_train <- ifelse(train_probs > 0.5, 1, 0)
pred_test <- ifelse(test_probs > 0.5, 1, 0)
pred_train_factor <- factor(pred_train, levels = c(0, 1))
train_label_factor <- factor(as.numeric(as.character(trainset$Personal.Loan)), levels = c(0, 1))
pred_test_factor <- factor(pred_test, levels = c(0, 1))
test_label_factor <- factor(as.numeric(as.character(testset$Personal.Loan)), levels = c(0, 1))
train_cm <- confusionMatrix(pred_train_factor, train_label_factor, positive = "1")
test_cm <- confusionMatrix(pred_test_factor, test_label_factor, positive = "1")
cat("Confusion Matrix (Training):\n")
print(train_cm)
cat("\nConfusion Matrix (Test):\n")
print(test_cm)

# --- Support Vector Machine (SVM) ---
# Convert target variable for SVM: "Yes" for approved, "No" for not
trainset$Personal.Loan <- factor(trainset$Personal.Loan, levels = c(1, 0))
testset$Personal.Loan <- factor(testset$Personal.Loan, levels = c(1, 0))

# Verify distribution (to avoid single-level issue)
table(trainset$Personal.Loan)
table(testset$Personal.Loan)

# caret requires valid labels, let's explicitly name them as characters:
levels(trainset$Personal.Loan) <- c("Yes", "No")
levels(testset$Personal.Loan) <- c("Yes", "No")

# Define the control for caret
ctrl <- caret::trainControl(method = "repeatedcv",
                            number = 5,
                            repeats = 2,
                            classProbs = TRUE,
                            summaryFunction = twoClassSummary)

# SVM grid for tuning parameter
svmGrid <- expand.grid(C = c(0.01, 0.1, 1, 10, 100))

# Train the linear SVM model (without prob.model)
svm_linear_model <- caret::train(Personal.Loan ~ ., data = trainset,
                                 method = "svmLinear",
                                 trControl = ctrl,
                                 metric = "ROC",
                                 tuneGrid = svmGrid,
                                 preProc = c("center", "scale"))

# Print results
print(svm_linear_model)

# Plot ROC vs Cost parameter C
library(ggplot2)
ggplot(svm_linear_model) + scale_x_log10()
# Generate predictions
svm_preds <- predict(svm_linear_model, newdata = testset)
svm_probs <- predict(svm_linear_model, newdata = testset, type = "prob")$Yes  

confusionMatrix(data = svm_preds, reference = testset$Personal.Loan, positive = "Yes")

svm_roc <- roc(response = testset$Personal.Loan, predictor = svm_probs, levels = c("No", "Yes"))
svm_auc <- auc(svm_roc)
cat("SVM AUC:", svm_auc, "\n")
# --- Decision Tree using rpart ---
treeGrid <- expand.grid(cp = seq(0.0001, 0.01, by = 0.001))
tree_model <- caret::train(Personal.Loan ~ ., data = trainset,
                           method = "rpart",
                           trControl = ctrl,
                           tuneGrid = treeGrid,
                           metric = "ROC")

print(tree_model$finalModel)
rpart.plot(tree_model$finalModel, yesno = 2, type = 2, extra = 0)

tree_preds <- predict(tree_model, newdata = testset)
tree_probs <- predict(tree_model, newdata = testset, type = "prob")$Yes

caret::confusionMatrix(data = tree_preds, reference = testset$Personal.Loan, positive = "Yes")

tree_roc <- pROC::roc(testset$Personal.Loan, tree_probs, levels = c("No", "Yes"))
tree_auc <- pROC::auc(tree_roc)

cat("Decision Tree AUC:", tree_auc, "\n")

# --- Bootstrap Logistic Regression ---
train_indices <- createDataPartition(df$Personal.Loan, p = 0.7, list = FALSE)
trainset <- df[train_indices, ]
testset <- df[-train_indices, ]
trainset$Personal.Loan <- factor(ifelse(trainset$Personal.Loan == 1, "Yes", "No"), levels = c("Yes", "No"))
testset$Personal.Loan <- factor(ifelse(testset$Personal.Loan == 1, "Yes", "No"), levels = c("Yes", "No"))
print(table(trainset$Personal.Loan, useNA = "ifany"))
train_control_boot <- trainControl(method = "boot", number = 50,
                                   classProbs = TRUE,
                                   summaryFunction = twoClassSummary,
                                   savePredictions = "final")
logit_boot <- caret::train(Personal.Loan ~ ., data = trainset,
                           method = "glm",
                           family = "binomial",
                           metric = "ROC",
                           trControl = train_control_boot)
print(logit_boot)

# --- Random Forest ---
rf_grid <- expand.grid(mtry = c(2, 3, 4, 5))
rf_model <- caret::train(Personal.Loan ~ ., data = trainset,
                         method = "rf",
                         trControl = ctrl,
                         metric = "ROC",
                         tuneGrid = rf_grid,
                         ntree = 500)
print(rf_model)
rf_preds <- predict(rf_model, newdata = testset)
rf_probs <- predict(rf_model, newdata = testset, type = "prob")$Yes
confusionMatrix(rf_preds, testset$Personal.Loan, positive = "Yes")
rf_roc <- roc(testset$Personal.Loan, rf_probs, levels = c("No", "Yes"))
rf_auc <- auc(rf_roc)
cat("Random Forest AUC:", rf_auc, "\n")
ggplot(rf_model)

# --- Deep Learning with Keras ---

rec <- recipe(Personal.Loan ~ ., data = df) %>%
  step_dummy(all_nominal(), -all_outcomes()) %>% 
  step_center(all_numeric(), -all_outcomes()) %>%
  step_scale(all_numeric(), -all_outcomes()) %>%
  prep(training = df)

df_processed <- bake(rec, new_data = df)
df_processed$Personal.Loan <- ifelse(df_processed$Personal.Loan == "1", 1, 0)

# Train/test split
set.seed(123)
train_indices <- createDataPartition(df_processed$Personal.Loan, p = 0.7, list = FALSE)
train_processed <- df_processed[train_indices, ]
test_processed <- df_processed[-train_indices, ]

# Convert to matrices
x_train <- as.matrix(train_processed[, setdiff(names(train_processed), "Personal.Loan")])
y_train <- matrix(train_processed$Personal.Loan, ncol=1)
x_test <- as.matrix(test_processed[, setdiff(names(test_processed), "Personal.Loan")])
y_test <- matrix(test_processed$Personal.Loan, ncol=1)

# Print dimensions to verify
print(dim(x_train))
print(dim(y_train))

# Create a simpler model with fewer layers
model <- keras_model_sequential()
model$add(layer_dense(units=16, activation='relu', input_shape=ncol(x_train)))
model$add(layer_dense(units=8, activation='relu'))
model$add(layer_dense(units=1, activation='sigmoid'))

# Compile with the strictly correct format for newer versions
model$compile(
  loss = 'binary_crossentropy',
  optimizer = 'adam',
  metrics = list(tf$keras$metrics$BinaryAccuracy())  # Use TensorFlow's keras metrics
)

# Evaluate model
scores <- model$evaluate(
  x = x_test,
  y = y_test,
  verbose = 0
)

cat("Test Loss:", scores[1], "Test Accuracy:", scores[2], "\n")

# Predict probabilities
pred_probs <- model$predict(
  x = x_test,
  verbose = 0
)

# Calculate ROC and AUC
roc_obj <- roc(y_test, as.vector(pred_probs))
auc_val <- auc(roc_obj)
cat("Test AUC:", auc_val, "\n")

# Plot ROC curve
plot(roc_obj, main = paste("ROC Curve - Neural Network (AUC =", round(auc_val, 3), ")"))


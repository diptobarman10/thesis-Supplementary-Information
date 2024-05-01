# Load required libraries
library(dplyr)

# Calculate summary statistics for demographic variables and personality traits
summary_stats <- Dataset %>%
  select(Gender, Age, Education, Social_media_use, Political_Label, CRT, Openness, Extraversion, Neuroticism, Agreeableness, Conscientiousness) %>%
  summarise_all(list(min = min, max = max, mean = mean, median = median, sd = sd))

print(summary_stats)

Dataset$Gender <- factor(Dataset$Gender, levels = c(1,2,3,4), labels = c('male', 'female','binary', 'prefer not to say'))
Dataset$Age <-factor(Dataset$Age, levels = c(1,2,3,4,5,6), labels = c('18-25', '26-35','36-45','46-55','56-65','65+'))
Dataset$Education <- factor(Dataset$Education, levels = c(1,2,3,4,5), labels = c('Less than high school', 'high school', 'undergraduate','graduate','postgraduate'))
Dataset$Social_media_use <- factor(Dataset$Social_media_use, levels = c(1,2,3,4,5), labels = c('0-1 hour','2-3 hours','4-5 hours','5-6 hours','+6 hours'))
Dataset$Political_Label <- factor(Dataset$Political_Label, levels = c(1,2,3,4,5), labels = c('extremely Liberal','moderately Liberal', 'neither liberal nor conservative', 'moderately conservative','extremely conservative'))


# Load required libraries
library(ggplot2)
library(gridExtra)

# Function to create bar plots for categorical variables
create_bar_plot <- function(data, var_name, title) {
  ggplot(data, aes(x = !!sym(var_name))) +
    geom_bar() +
    ggtitle(title) +
    theme_minimal()
}

# Function to create histograms for continuous variables
create_histogram <- function(data, var_name, title) {
  ggplot(data, aes(x = !!sym(var_name))) +
    geom_histogram(binwidth = 1) +
    ggtitle(title) +
    theme_minimal()
}

# Create plots for categorical variables
gender_plot <- create_bar_plot(Dataset, "Gender", "Gender Distribution")
age_plot <- create_bar_plot(Dataset, "Age", "Age Distribution")
education_plot <- create_bar_plot(Dataset, "Education", "Education Distribution")
social_media_use_plot <- create_bar_plot(Dataset, "Social_media_use", "Social Media Use Distribution")
political_label_plot <- create_bar_plot(Dataset, "Political_Label", "Political Label Distribution")
crt_plot <- create_bar_plot(Dataset, "CRT", "CRT Distribution")

# Create plots for continuous variables
openness_plot <- create_histogram(Dataset, "Openness", "Openness Distribution")
extraversion_plot <- create_histogram(Dataset, "Extraversion", "Extraversion Distribution")
neuroticism_plot <- create_histogram(Dataset, "Neuroticism", "Neuroticism Distribution")
agreeableness_plot <- create_histogram(Dataset, "Agreeableness", "Agreeableness Distribution")
conscientiousness_plot <- create_histogram(Dataset, "Conscientiousness", "Conscientiousness Distribution")

# Arrange the plots in a grid
grid.arrange(gender_plot, age_plot, education_plot, social_media_use_plot, political_label_plot,
             crt_plot, openness_plot, extraversion_plot, neuroticism_plot, agreeableness_plot, conscientiousness_plot,
             ncol = 3)

# Load required libraries
library(ggplot2)
library(gridExtra)

# Function to create Q-Q plots
create_qq_plot <- function(data, var_name, title) {
  ggplot(data, aes(sample = !!sym(var_name))) +
    stat_qq() +
    stat_qq_line() +
    ggtitle(title) +
    theme_minimal()
}

# Variables for each stimulus type
fake_news_accuracy <- c("Fake_news_A", "Fake_news_Flag_A", "Fake_news_W_A", "True_news_A")
fake_news_sharing <- c("Fake_news_S", "Fake_news_Flag_S", "Fake_news_W_S", "True_news_S")
fake_news_trustworthiness <- c("Fake_news_Flag_T", "Fake_news_W_T")

# Check normality for accuracy ratings
accuracy_normality_tests <- lapply(fake_news_accuracy, function(var_name) {
  shapiro.test(Dataset[[var_name]])
})
names(accuracy_normality_tests) <- fake_news_accuracy
print(accuracy_normality_tests)

# Q-Q plots for accuracy ratings
accuracy_qq_plots <- lapply(fake_news_accuracy, function(var_name) {
  create_qq_plot(Dataset, var_name, paste("Q-Q plot for", var_name, "Accuracy Ratings"))
})
do.call(grid.arrange, c(accuracy_qq_plots, ncol = 2))

# Check normality for sharing likelihood ratings
sharing_normality_tests <- lapply(fake_news_sharing, function(var_name) {
  shapiro.test(Dataset[[var_name]])
})
names(sharing_normality_tests) <- fake_news_sharing
print(sharing_normality_tests)

# Q-Q plots for sharing likelihood ratings
sharing_qq_plots <- lapply(fake_news_sharing, function(var_name) {
  create_qq_plot(clean_dataset, var_name, paste("Q-Q plot for", var_name, "Sharing Likelihood Ratings"))
})
do.call(grid.arrange, c(sharing_qq_plots, ncol = 2))

# Check normality for trustworthiness ratings
trustworthiness_normality_tests <- lapply(fake_news_trustworthiness, function(var_name) {
  shapiro.test(Dataset[[var_name]])
})
names(trustworthiness_normality_tests) <- fake_news_trustworthiness
print(trustworthiness_normality_tests)

# Q-Q plots for trustworthiness ratings
trustworthiness_qq_plots <- lapply(fake_news_trustworthiness, function(var_name) {
  create_qq_plot(clean_dataset, var_name, paste("Q-Q plot for", var_name, "Trustworthiness Ratings"))
})
do.call(grid.arrange, c(trustworthiness_qq_plots, ncol = 2))

# Load required library
library(FSA)

# Friedman test for accuracy ratings
accuracy_friedman <- friedman.test(as.matrix(Dataset[, fake_news_accuracy]))
cat("Friedman test for accuracy ratings:\n")
print(accuracy_friedman)

# Friedman test for sharing likelihood ratings
sharing_friedman <- friedman.test(as.matrix(clean_dataset[, fake_news_sharing]))
cat("\nFriedman test for sharing likelihood ratings:\n")
print(sharing_friedman)

trust_friedman <- friedman.test(as.matrix(clean_dataset[, fake_news_trustworthiness]))
cat("\nFriedman test for trust likelihood ratings:\n")
print(trust_friedman)

-----
# Pairwise comparison using Wilcoxon signed-rank test for Fake_news_A and Fake_news_Flag_A
test <- wilcox.test(clean_dataset[["Fake_news_A"]], clean_dataset[["Fake_news_Flag_A"]], paired = TRUE)
p_value <- test$p.value

# Bonferroni correction
bonferroni_alpha <- 0.05 / 1  # Since we're only doing one comparison
adjusted_p_value <- p.adjust(p_value, method = "bonferroni")

cat("Pairwise Wilcoxon signed-rank test with Bonferroni correction for Fake_news_A vs. Fake_news_Flag_A:\n")
cat("Original p-value:", p_value, "\n")
cat("Adjusted p-value:", adjusted_p_value, "\n")

------
# Pairwise comparison using Wilcoxon signed-rank test for Fake_news_W_T and Fake_news_Flag_T
test <- wilcox.test(clean_dataset[["Fake_news_W_T"]], clean_dataset[["Fake_news_Flag_T"]], paired = TRUE)
p_value <- test$p.value

# Bonferroni correction
bonferroni_alpha <- 0.05 / 1  # Since we're only doing one comparison
adjusted_p_value <- p.adjust(p_value, method = "bonferroni")

cat("Pairwise Wilcoxon signed-rank test with Bonferroni correction for Fake_news_W_T vs. Fake_news_Flag_T:\n")
cat("Original p-value:", p_value, "\n")
cat("Adjusted p-value:", adjusted_p_value, "\n")

----
# Pairwise comparisons using Wilcoxon signed-rank test for accuracy ratings
accuracy_pairs <- combn(fake_news_accuracy, 2)
pairwise_wilcox_results <- apply(accuracy_pairs, 2, function(pair) {
  test <- wilcox.test(Dataset[[pair[1]]], Dataset[[pair[2]]], paired = TRUE)
  p_value <- test$p.value
  adjusted_p_value <- p.adjust(p_value, method = "bonferroni", n = choose(length(fake_news_accuracy), 2))
  return(c(pair[1], pair[2], p_value, adjusted_p_value))
})
pairwise_wilcox_df <- as.data.frame(t(pairwise_wilcox_results))
colnames(pairwise_wilcox_df) <- c("Stimulus 1", "Stimulus 2", "p-value", "adjusted_p_value")

# Print the results
print(pairwise_wilcox_df)

-----
# Pairwise comparisons using Wilcoxon signed-rank test for sharing ratings
accuracy_pairs <- combn(fake_news_sharing, 2)
pairwise_wilcox_results <- apply(accuracy_pairs, 2, function(pair) {
  test <- wilcox.test(clean_dataset[[pair[1]]], clean_dataset[[pair[2]]], paired = TRUE)
  p_value <- test$p.value
  adjusted_p_value <- p.adjust(p_value, method = "bonferroni", n = choose(length(fake_news_sharing), 2))
  return(c(pair[1], pair[2], p_value, adjusted_p_value))
})
pairwise_wilcox_df <- as.data.frame(t(pairwise_wilcox_results))
colnames(pairwise_wilcox_df) <- c("Stimulus 1", "Stimulus 2", "p-value", "adjusted_p_value")

# Print the results
print(pairwise_wilcox_df)

----
# Pairwise comparisons using Wilcoxon signed-rank test for trust ratings
accuracy_pairs <- combn(fake_news_trustworthiness, 2)
pairwise_wilcox_results <- apply(accuracy_pairs, 2, function(pair) {
  test <- wilcox.test(clean_dataset[[pair[1]]], clean_dataset[[pair[2]]], paired = TRUE)
  p_value <- test$p.value
  adjusted_p_value <- p.adjust(p_value, method = "bonferroni", n = choose(length(fake_news_trustworthiness), 2))
  return(c(pair[1], pair[2], p_value, adjusted_p_value))
})
pairwise_wilcox_df <- as.data.frame(t(pairwise_wilcox_results))
colnames(pairwise_wilcox_df) <- c("Stimulus 1", "Stimulus 2", "p-value", "adjusted_p_value")

# Print the results
print(pairwise_wilcox_df)
---
# Load the ggplot2 package
library(ggplot2)

# Create a long format data frame for sharing ratings
sharing_long_df <- reshape2::melt(Dataset, id.vars = NULL, measure.vars = fake_news_sharing, variable.name = "Stimulus", value.name = "Sharing")

# Define custom y-axis labels for sharing
y_sharing_labels <- c("1" = "Extremely unlikely",
              "2" = "Somewhat unlikely",
              "3" = "Neither likely nor unlikely",
              "4" = "Somewhat likely",
              "5" = "Extremely likely")

# Define custom y-axis labels for accuracy 
y_accuracy_labels <- c("1" = "Not accurate at all",
                      "2" = "somewhat inaccurate",
                      "3" = "Neither inaccurate nor accurate",
                      "4" = "somewhat accurate",
                      "5" = "very accurate")

# Define custom y-axis labels for accuracy 
y_trust_labels <- c("1" = "Very untrustworthy",
                       "2" = "somewhat untrustworthy",
                       "3" = "Neither untrustworthy nor trustworthy",
                       "4" = "somewhat trustworthy",
                       "5" = "very trustworthy")

# Create a box plot for sharing ratings
ggplot(sharing_long_df, aes(x = Stimulus, y = Sharing)) +
  geom_boxplot() +
  scale_y_continuous(breaks = 1:5, labels = y_sharing_labels) +
  labs(title = "Box Plot of Sharing Ratings", x = "Stimulus", y = "Sharing Ratings")

# Create a violin plot for sharing ratings
ggplot(sharing_long_df, aes(x = Stimulus, y = Sharing)) +
  geom_violin() + 
  scale_y_continuous(breaks = 1:5, labels = y_sharing_labels)+
  labs(title = "Violin Plot of Sharing Ratings", x = "Stimulus", y = "Sharing Ratings")

---
  
# Create a long format data frame for accuracy ratings
  accuracy_long_df <- reshape2::melt(Dataset, id.vars = NULL, measure.vars = fake_news_accuracy, variable.name = "Stimulus", value.name = "accuracy")

# Create a box plot for sharing ratings
ggplot(accuracy_long_df, aes(x = Stimulus, y = accuracy)) +
  geom_boxplot() +
  scale_y_continuous(breaks = 1:5, labels = y_accuracy_labels)+
  labs(title = "Box Plot of accuracy Ratings", x = "Stimulus", y = "accuracy Ratings")

# Create a violin plot for sharing ratings
ggplot(accuracy_long_df, aes(x = Stimulus, y = accuracy)) +
  geom_violin() +
  scale_y_continuous(breaks = 1:5, labels = y_accuracy_labels)+
  labs(title = "Violin Plot of accuracy Ratings", x = "Stimulus", y = "accuracy Ratings")

----

# Create a long format data frame for trust ratings
trust_long_df <- reshape2::melt(Dataset, id.vars = NULL, measure.vars = fake_news_trustworthiness, variable.name = "Stimulus", value.name = "trust")

# Create a box plot for sharing ratings
ggplot(trust_long_df, aes(x = Stimulus, y = trust)) +
  geom_boxplot() +
  scale_y_continuous(breaks = 1:5, labels = y_trust_labels)+
  labs(title = "Box Plot of trust Ratings", x = "Stimulus", y = "trust Ratings")

# Create a violin plot for sharing ratings
ggplot(trust_long_df, aes(x = Stimulus, y = trust)) +
  geom_violin() +
  scale_y_continuous(breaks = 1:5, labels = y_trust_labels)+
  labs(title = "Violin Plot of trust Ratings", x = "Stimulus", y = "trust Ratings")

----
# Define the variables to compute medians for
variable_names <- c(fake_news_trustworthiness, fake_news_accuracy, fake_news_sharing)

# Compute the median value for each variable
median_values <- sapply(variable_names, function(variable) {
  median(clean_dataset[[variable]], na.rm = TRUE)
})

# Create a data frame to display the results
print(median_values)

----
  # Scatterplot matrix for accuracy variables
  accuracy_vars <- c("Fake_news_A", "Fake_news_Flag_A", "Fake_news_W_A", "True_news_A")
pairs(clean_dataset[, accuracy_vars], main = "Scatterplot Matrix for Accuracy Variables")

---
# Scatterplot matrix for sharing variables
sharing_vars <- c("Fake_news_S", "Fake_news_Flag_S", "Fake_news_W_S", "True_news_S")
pairs(clean_dataset[, sharing_vars], main = "Scatterplot Matrix for sharing Variables")

---
  
trust_var <- c("Fake_news_Flag_T", "Fake_news_W_T")
pairs(clean_dataset[, trust_var], main = "Scatterplot Matrix for Trust Variables", pch = 3)

-----
# Pearson's correlation matrix for accuracy variables
pearson_cor_matrix <- cor(clean_dataset[, accuracy_vars], method = "pearson")
cat("Pearson's Correlation Matrix for Accuracy Variables:\n")
print(pearson_cor_matrix)

# Spearman's rank correlation matrix for accuracy variables (if needed)
spearman_cor_matrix <- cor(clean_dataset[, accuracy_vars], method = "spearman")
cat("Spearman's Rank Correlation Matrix for Accuracy Variables:\n")
print(spearman_cor_matrix)

----
  # Function to calculate correlation and p-value
  correlation_test <- function(var1, var2, method = "spearman") {
    test <- cor.test(clean_dataset[[var1]], clean_dataset[[var2]], method = method)
    return(c(var1, var2, test$estimate, test$p.value))
  }

# Pairwise correlation tests for accuracy variables
accuracy_vars <- c("Fake_news_A", "Fake_news_Flag_A", "Fake_news_W_A", "True_news_A")
accuracy_pairs <- combn(accuracy_vars, 2)
correlation_results <- apply(accuracy_pairs, 2, function(pair) {
  correlation_test(pair[1], pair[2], method = "spearman") # Change method to "person" if needed
})

# Create a data frame to display the results
correlation_df <- as.data.frame(t(correlation_results))
colnames(correlation_df) <- c("Variable 1", "Variable 2", "Correlation", "p-value")
# Print the results
print(correlation_df)
-----

  ## Kruskal-Wallis test for Predicting variables ---- For Fake_news_A

 # Gender
kruskal_gender <- kruskal.test(Fake_news_A ~ Gender, data = clean_dataset)
cat("Kruskal-Wallis test for Gender:\n")
print(kruskal_gender)

# Education
kruskal_education <- kruskal.test(Fake_news_A ~ Education, data = clean_dataset)
cat("Kruskal-Wallis test for Education:\n")
print(kruskal_education)

# Social Media Use
kruskal_social_media_use <- kruskal.test(Fake_news_A ~ Social_media_use, data = clean_dataset)
cat("Kruskal-Wallis test for Social Media Use:\n")
print(kruskal_social_media_use)

# Political Label
kruskal_political_label <- kruskal.test(Fake_news_A ~ Political_Label, data = clean_dataset)
cat("Kruskal-Wallis test for Political Label:\n")
print(kruskal_political_label)

# CRT
kruskal_CRT <- kruskal.test(Fake_news_A ~ clean_dataset$CRT, data = clean_dataset)
cat("Kruskal-Wallis test for Political Label:\n")
print(kruskal_CRT)

#Age
kruskal_Age <- kruskal.test(Fake_news_A ~ Age, data = clean_dataset)
cat("Kruskal-Wallis test for Political Label:\n")
print(kruskal_Age)

# Independent variables
independent_vars <- c("Gender", "Education", "Social_media_use", "Political_Label", "Age", "CRT", "Openness", "Extraversion", "Neuroticism", "Agreeableness", "Conscientiousness")

# Dependent variables
dependent_vars <- c("Fake_news_A", "Fake_news_Flag_A", "Fake_news_W_A", "True_news_A",
                    "Fake_news_S", "Fake_news_Flag_S", "Fake_news_W_S", "True_news_S",
                    "Fake_news_Flag_T", "Fake_news_W_T")

# Perform Kruskal-Wallis tests for all values..
for (dep_var in dependent_vars) {
  cat("\nKruskal-Wallis tests for", dep_var, ":\n")
  
  for (ind_var in independent_vars) {
    kruskal_result <- kruskal.test(as.formula(paste(dep_var, "~", ind_var)), data = clean_dataset)
    cat(ind_var, ": Chi-squared =", kruskal_result$statistic, ", p-value =", kruskal_result$p.value, "\n")
  }
}

# only the ones that are significant

# Perform Kruskal-Wallis tests
for (dep_var in dependent_vars) {
  cat("\nKruskal-Wallis tests for", dep_var, ":\n")
  
  for (ind_var in independent_vars) {
    kruskal_result <- kruskal.test(as.formula(paste(dep_var, "~", ind_var)), data = clean_dataset)
    p_value <- kruskal_result$p.value
    
    if (p_value < 0.05) {
      cat(ind_var, ": Chi-squared =", kruskal_result$statistic, ", p-value =", p_value, "\n")
    }
  }
}

-----

  # List of continuous independent variables
  independent_vars <- c("Openness", "Extraversion", "Neuroticism", "Agreeableness", "Conscientiousness")

# List of continuous dependent variables (replace with your actual variable names)
dependent_vars <- c("Fake_news_A", "Fake_news_Flag_A", "Fake_news_W_A", "True_news_A",
                    "Fake_news_S", "Fake_news_Flag_S", "Fake_news_W_S", "True_news_S",
                    "Fake_news_Flag_T", "Fake_news_W_T")

# Initialize an empty data frame to store the results
results <- data.frame()

# Loop through the dependent and independent variables
for (dep_var in dependent_vars) {
  for (ind_var in independent_vars) {
    # Perform Spearman's rank correlation
    correlation <- cor.test(clean_dataset[[dep_var]], clean_dataset[[ind_var]], method = "spearman")
    
    # Store the results in a temporary data frame
    temp_df <- data.frame(Dependent = dep_var,
                          Independent = ind_var,
                          Correlation = correlation$estimate,
                          p_value = correlation$p.value)
    
    # Append the temporary data frame to the results data frame
    results <- rbind(results, temp_df)
  }
}

# Print the results
print(results)


----- ## NEW IMPLEMENTATION

  # Load necessary libraries
  library(tidyr)
library(dplyr)

# Independent variables (Likert scale and continuous)
iv_likert <- c("Education", "Social_media_use", "Political_Label", "Age", "CRT")
iv_continuous <- c("Openness", "Extraversion", "Neuroticism", "Agreeableness", "Conscientiousness")
# Combine independent variables
iv_all <- c(iv_likert, iv_continuous)

# Dependent variables
dv <- c("Fake_news_A", "Fake_news_Flag_A", "Fake_news_W_A", "True_news_A", "Fake_news_S", "Fake_news_Flag_S", "Fake_news_W_S", "True_news_S", "Fake_news_Flag_T", "Fake_news_W_T")

# Convert relevant columns to numeric format
clean_dataset[iv_all] <- lapply(clean_dataset[iv_all], as.numeric)
clean_dataset[dv] <- lapply(clean_dataset[dv], as.numeric)
clean_dataset[iv_likert] <- lapply(clean_dataset[iv_likert], as.numeric)

# Initialize an empty data frame to store the results
results <- data.frame()

# Loop through the dependent and independent variables
for (dep_var in dv) {
  for (ind_var in iv_all) {
    # Perform Spearman's rank correlation
    correlation <- cor.test(clean_dataset[[dep_var]], clean_dataset[[ind_var]], method = "spearman")
    
    # Store the results in a temporary data frame
    temp_df <- data.frame(Dependent = dep_var,
                          Independent = ind_var,
                          Correlation = correlation$estimate,
                          p_value = correlation$p.value)
    
    # Append the temporary data frame to the results data frame
    results <- rbind(results, temp_df)
  }
}
# Print the results
print(results)


-----
  # Initialize an empty data frame to store the results
  results <- data.frame()
  
  # Loop through the dependent and independent variables
  for (dep_var in dv) {
    for (ind_var in iv_all) {
      # Perform Spearman's rank correlation
      correlation <- cor.test(clean_dataset[[dep_var]], clean_dataset[[ind_var]], method = "spearman")
      
      # Check if the p-value is less than 0.05
      if (correlation$p.value < 0.05) {
        # Store the results in a temporary data frame
        temp_df <- data.frame(Dependent = dep_var,
                              Independent = ind_var,
                              Correlation = correlation$estimate,
                              p_value = correlation$p.value)
        
        # Append the temporary data frame to the results data frame
        results <- rbind(results, temp_df)
      }
    }
  }
# Print the results
print(results)

------ ## Visuallising the plots

  # Install ggplot2 if you haven't already
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    install.packages("ggplot2")
  }

# Load the ggplot2 package
library(ggplot2)

# Create a scatter plot of the relationship between two variables
neuroticism_Fake_new_W_T <- ggplot(data = clean_dataset, aes(x = Neuroticism, y = clean_dataset$Fake_news_W_T)) +
  geom_point() +
  labs(title = "Scatter plot of Neuroticism and Fake_news_W_T",
       x = "Neuroticism",
       y = "Fake_news_W_T") +
  theme_minimal()

# Create a scatter plot of the relationship between two variables
neuroticism_Fake_new_Flag_T <- ggplot(data = clean_dataset, aes(x = Neuroticism, y = clean_dataset$Fake_news_Flag_T)) +
  geom_point() +
  labs(title = "Scatter plot of Neuroticism and Fake_news_W_T",
       x = "Neuroticism",
       y = "Fake_news_Flag_T") +
  theme_minimal()
# Arrange the plots in a grid
grid.arrange(neuroticism_Fake_new_W_T, neuroticism_Fake_new_Flag_T,
             ncol = 2)

---- ## Predictions
  
install.packages("pls")
install.packages("loadings")

library(pls)

# Assuming your data is in a data frame called 'data'
# Replace 'dependent_variable' with the actual column name of the dependent variable
# Replace 'independent_variable_1', 'independent_variable_2', etc. with the actual column names of the independent variables
pls_model <- plsr(clean_dataset$Fake_news_W_T ~ Openness + Extraversion + Neuroticism + Agreeableness + Conscientiousness + CRT + Age + Political_Label + Social_media_use + Education $, data = clean_dataset, validation = "CV")

summary(pls_model)
# Load the required library

# Install and load the mixOmics package
install.packages("mixOmics")
library(mixOmics)

# Calculate VIP scores
vip_scores <- vip(pls_model)

# Print the VIP scores
print(vip_scores)

#### Using gneralised Linear models.

glm_model <- glm(Fake_news_W_T ~ Openness + Extraversion + Neuroticism + Agreeableness + Conscientiousness + CRT + Age + Political_Label + Social_media_use + Education, data = clean_dataset)
summary(glm_model)

glm_model1 <- glm(Fake_news_Flag_T ~ Openness + Extraversion + Neuroticism + Agreeableness + Conscientiousness + CRT + Age + Political_Label + Social_media_use + Education, data = clean_dataset)
summary(glm_model1)

# Install and load the package
install.packages("randomForest")
library(randomForest)

rf_model <- randomForest(Fake_news_W_T ~ Openness + Extraversion + Neuroticism + Agreeableness + Conscientiousness + CRT + Age + Political_Label + Social_media_use + Education, data = clean_dataset)
summary(rf_model)
importance <- importance(rf_model)
print(importance)
# Install and load the package
install.packages("gbm")
library(gbm)

gbm_model <- gbm(Fake_news_W_T ~ Openness + Extraversion + Neuroticism + Agreeableness + Conscientiousness + CRT + Age + Political_Label + Social_media_use + Education, data = clean_dataset, distribution = "gaussian", n.trees = 100, interaction.depth = 4, shrinkage = 0.01)
summary(gbm_model)

# Install and load the package
install.packages("e1071")
library(e1071)

svm_model <- svm(Fake_news_W_T ~ Openness + Extraversion + Neuroticism + Agreeableness + Conscientiousness + CRT + Age + Political_Label + Social_media_use + Education, data = clean_dataset, type = "eps-regression")
summary(svm_model)








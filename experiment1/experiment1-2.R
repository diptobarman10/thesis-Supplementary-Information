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

# Check normality for sharing likelihood ratings
sharing_normality_tests <- lapply(fake_news_sharing, function(var_name) {
  shapiro.test(Dataset[[var_name]])
})
names(sharing_normality_tests) <- fake_news_sharing
print(sharing_normality_tests)

# Check normality for trustworthiness ratings
trustworthiness_normality_tests <- lapply(fake_news_trustworthiness, function(var_name) {
  shapiro.test(Dataset[[var_name]])
})
names(trustworthiness_normality_tests) <- fake_news_trustworthiness
print(trustworthiness_normality_tests)

# Load required library
library(FSA)

# Friedman test for accuracy ratings
accuracy_friedman <- friedman.test(as.matrix(Dataset[, fake_news_accuracy]))
cat("Friedman test for accuracy ratings:\n")
print(accuracy_friedman)

# Friedman test for sharing likelihood ratings
sharing_friedman <- friedman.test(as.matrix(Dataset[, fake_news_sharing]))
cat("\nFriedman test for sharing likelihood ratings:\n")
print(sharing_friedman)

trust_friedman <- friedman.test(as.matrix(Dataset[, fake_news_trustworthiness]))
cat("\nFriedman test for trust likelihood ratings:\n")
print(trust_friedman)

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

# Pairwise comparisons using Wilcoxon signed-rank test for sharing ratings
accuracy_pairs <- combn(fake_news_sharing, 2)
pairwise_wilcox_results <- apply(accuracy_pairs, 2, function(pair) {
  test <- wilcox.test(Dataset[[pair[1]]], Dataset[[pair[2]]], paired = TRUE)
  p_value <- test$p.value
  adjusted_p_value <- p.adjust(p_value, method = "bonferroni", n = choose(length(fake_news_sharing), 2))
  return(c(pair[1], pair[2], p_value, adjusted_p_value))
})
pairwise_wilcox_df <- as.data.frame(t(pairwise_wilcox_results))
colnames(pairwise_wilcox_df) <- c("Stimulus 1", "Stimulus 2", "p-value", "adjusted_p_value")

# Print the results
print(pairwise_wilcox_df)

# Pairwise comparisons using Wilcoxon signed-rank test for trust ratings
accuracy_pairs <- combn(fake_news_trustworthiness, 2)
pairwise_wilcox_results <- apply(accuracy_pairs, 2, function(pair) {
  test <- wilcox.test(Dataset[[pair[1]]], Dataset[[pair[2]]], paired = TRUE)
  p_value <- test$p.value
  adjusted_p_value <- p.adjust(p_value, method = "bonferroni", n = choose(length(fake_news_trustworthiness), 2))
  return(c(pair[1], pair[2], p_value, adjusted_p_value))
})
pairwise_wilcox_df <- as.data.frame(t(pairwise_wilcox_results))
colnames(pairwise_wilcox_df) <- c("Stimulus 1", "Stimulus 2", "p-value", "adjusted_p_value")

# Print the results
print(pairwise_wilcox_df)


# Spearman's rank correlation matrix for accuracy variables (if needed)
spearman_cor_matrix <- cor(Dataset[, fake_news_accuracy], method = "spearman")
cat("Spearman's Rank Correlation Matrix for Accuracy Variables:\n")
print(spearman_cor_matrix)

# Install the package if not already installed
if(!require(Hmisc)) {
  install.packages("Hmisc")
}

# Load the package
library(Hmisc)

# Assuming your data frame is named numeric_data
result <- rcorr(as.matrix(sapply(numeric_data, as.numeric)), type = "spearman")

print(result)

# Extract the correlation matrix and p-value matrix
cor_matrix <- result$r
p_matrix <- result$P

# Create an empty matrix of the same size to store asterisks
ast_matrix <- matrix("", nrow = nrow(cor_matrix), ncol = ncol(cor_matrix))

# Mark significant correlations with asterisks
ast_matrix[p_matrix < 0.05] <- "*"

# Combine correlation matrix with asterisks
cor_table <- matrix(paste0(round(cor_matrix, 2), ast_matrix), nrow = nrow(cor_matrix))


# Convert to a data frame for easy viewing
df_cor_table <- as.data.frame(cor_table)
rownames(df_cor_table) <- rownames(cor_matrix)
colnames(df_cor_table) <- colnames(cor_matrix)

print(df_cor_table)

gt_table <- gt(df_cor_table)
print(gt_table)

install.packages("corrplot")
library(corrplot)

# Using cor_matrix from your code and assuming it's a matrix of correlations
corrplot(cor_matrix, method = "shade", type = "upper", 
         order = "original", addCoef.col = "black", 
         tl.col = "black", tl.srt = 45, 
         diag = FALSE, cl.lim = c(-1, 1),
         p.mat = p_matrix, sig.level = 0.05, insig = "blank")


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
    kruskal_result <- kruskal.test(as.formula(paste(dep_var, "~", ind_var)), data = df)
    cat(ind_var, ": Chi-squared =", kruskal_result$statistic, ", p-value =", kruskal_result$p.value, "\n")
  }
}

# only the ones that are significant
# Perform Kruskal-Wallis tests
for (dep_var in dependent_vars) {
  cat("\nKruskal-Wallis tests for", dep_var, ":\n")
  
  for (ind_var in independent_vars) {
    kruskal_result <- kruskal.test(as.formula(paste(dep_var, "~", ind_var)), data = df)
    p_value <- kruskal_result$p.value
    
    if (p_value < 0.05) {
      cat(ind_var, ": Chi-squared =", kruskal_result$statistic, ", p-value =", p_value, "\n")
    }
  }
}


## Correlations

# Load the necessary library
library(dplyr)

# Extract the names of all numeric variables
variable_names <- names(numeric_data)

# Initialize a data frame to store results
results <- data.frame(Var1 = character(), Var2 = character(), Correlation = numeric(), P_value = numeric(), stringsAsFactors = FALSE)

# Loop through each pair of variables
for(i in 1:(length(variable_names) - 1)) {
  for(j in (i + 1):length(variable_names)) {
    var1 <- variable_names[i]
    var2 <- variable_names[j]
    
    # Perform correlation test
    test_result <- cor.test(numeric_data[[var1]], numeric_data[[var2]], method = "spearman")
    
    # Store results
    results <- rbind(results, data.frame(Var1 = var1, Var2 = var2, Correlation = test_result$estimate, P_value = test_result$p.value))
  }
}

# Filter for significant correlations (e.g., p-value < 0.05)
significant_results <- results %>% filter(P_value < 0.05)

# View the significant results
print(significant_results)




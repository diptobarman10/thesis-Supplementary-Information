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


# Variables for each stimulus type
fake_news_accuracy <- c("Fake_news_A", "Fake_news_Flag_A", "Fake_news_W_A", "True_news_A")
fake_news_sharing <- c("Fake_news_S", "Fake_news_Flag_S", "Fake_news_W_S", "True_news_S")
fake_news_trustworthiness <- c("Fake_news_Flag_T", "Fake_news_W_T")

# Friedman test for accuracy ratings
accuracy_friedman <- friedman.test(as.matrix(Dataset[, fake_news_accuracy]))
cat("Friedman test for accuracy ratings:\n")
print(accuracy_friedman)

# Pairwise comparisons using Wilcoxon signed-rank test for accuracy ratings
accuracy_pairs <- combn(fake_news_accuracy, 2)
pairwise_wilcox_results <- apply(accuracy_pairs, 2, function(pair) {
  test <- wilcox.test(df[[pair[1]]], df[[pair[2]]], paired = TRUE)
  p_value <- test$p.value
  adjusted_p_value <- p.adjust(p_value, method = "bonferroni", n = choose(length(fake_news_accuracy), 2))
  return(c(pair[1], pair[2], p_value, adjusted_p_value))
})
pairwise_wilcox_df <- as.data.frame(t(pairwise_wilcox_results))
colnames(pairwise_wilcox_df) <- c("Stimulus 1", "Stimulus 2", "p-value", "adjusted_p_value")

# Print the results
print(pairwise_wilcox_df)

# Friedman test for sharing likelihood ratings
sharing_friedman <- friedman.test(as.matrix(Dataset[, fake_news_sharing]))
cat("\nFriedman test for sharing likelihood ratings:\n")
print(sharing_friedman)

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

trust_friedman <- friedman.test(as.matrix(Dataset[, fake_news_trustworthiness]))
cat("\nFriedman test for trust likelihood ratings:\n")
print(trust_friedman)

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


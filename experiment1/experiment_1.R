df1 <- read.csv("./Desktop/experiment1/Comparision/exp1.csv")
df2 <- read.csv('./Desktop/experiment1/Comparision/exp2.csv')

library(dplyr)

# Calculate summary statistics for demographic variables and personality traits
summary_stats1 <- df1 %>%
  select(Gender, Age, Education, Social_media_use, Political_Label, CRT, Openness, Extraversion, Neuroticism, Agreeableness, Conscientiousness) %>%
  summarise_all(list(min = min, max = max, mean = mean, median = median, sd = sd))

# Calculate summary statistics for demographic variables and personality traits
summary_stats2 <- df2 %>%
  select(Gender, Age, Education, Social_media_use, Political_Label, CRT, Openness, Extraversion, Neuroticism, Agreeableness, Conscientiousness) %>%
  summarise_all(list(min = min, max = max, mean = mean, median = median, sd = sd))

print(summary_stats1)
print(summary_stats2)

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

# Load required libraries
library(ggplot2)
library(gridExtra)

# Create plots for categorical variables
gender_plot1 <- create_bar_plot(df1, "Gender", "Gender Distribution")
age_plot1 <- create_bar_plot(df1, "Age", "Age Distribution")
education_plot1 <- create_bar_plot(df1, "Education", "Education Distribution")
social_media_use_plot1 <- create_bar_plot(df1, "Social_media_use", "Social Media Use Distribution")
political_label_plot1 <- create_bar_plot(df1, "Political_Label", "Political Label Distribution")
crt_plot1 <- create_bar_plot(df1, "CRT", "CRT Distribution")

# Create plots for continuous variables
openness_plot1 <- create_histogram(df1, "Openness", "Openness Distribution")
extraversion_plot1 <- create_histogram(df1, "Extraversion", "Extraversion Distribution")
neuroticism_plot1 <- create_histogram(df1, "Neuroticism", "Neuroticism Distribution")
agreeableness_plot1 <- create_histogram(df1, "Agreeableness", "Agreeableness Distribution")
conscientiousness_plot1 <- create_histogram(df1, "Conscientiousness", "Conscientiousness Distribution")

# Create plots for categorical variables
gender_plot2 <- create_bar_plot(df2, "Gender", "Gender Distribution")
age_plot2 <- create_bar_plot(df2, "Age", "Age Distribution")
education_plot2 <- create_bar_plot(df2, "Education", "Education Distribution")
social_media_use_plot2 <- create_bar_plot(df2, "Social_media_use", "Social Media Use Distribution")
political_label_plot2 <- create_bar_plot(df2, "Political_Label", "Political Label Distribution")
crt_plot2 <- create_bar_plot(df2, "CRT", "CRT Distribution")

# Create plots for continuous variables
openness_plot2 <- create_histogram(df2, "Openness", "Openness Distribution")
extraversion_plot2 <- create_histogram(df2, "Extraversion", "Extraversion Distribution")
neuroticism_plot2 <- create_histogram(df2, "Neuroticism", "Neuroticism Distribution")
agreeableness_plot2 <- create_histogram(df2, "Agreeableness", "Agreeableness Distribution")
conscientiousness_plot2 <- create_histogram(df2, "Conscientiousness", "Conscientiousness Distribution")

# Arrange the plots in a grid
grid.arrange(gender_plot1, gender_plot2, age_plot1, age_plot2, education_plot1,education_plot2, social_media_use_plot1,social_media_use_plot2, political_label_plot1,political_label_plot2,
             crt_plot1,crt_plot2,
             ncol = 2)


# Variables for each stimulus type
news_accuracy <- c("Fake_news_1_A","Fake_news_2_A", "Fake_news_Flag_1_A", "Fake_news_Flag_2_A","Fake_news_1_W_A","Fake_news_2_W_A", "Fake_news_3_W_A", "True_news_1_A","True_News_2_A","True_News_3_A")
news_sharing <- c("Fake_news_1_S","Fake_news_2_S", "Fake_news_Flag_1_S", "Fake_news_Flag_2_S","Fake_news_1_W_S","Fake_news_2_W_S", "Fake_news_3_W_S", "True_news_1_S","True_News_2_S","True_News_3_S")
news_trust <- c("Fake_news_Flag_1_T", "Fake_news_Flag_2_T", "Fake_news_1_W_T", "Fake_news_2_W_T", "Fake_news_3_W_T")

# Load the ggplot2 package
library(ggplot2)
# Create a long format data frame for sharing ratings
sharing_long_df <- reshape2::melt(clean_dataset, id.vars = NULL, measure.vars = news_sharing, variable.name = "Stimulus", value.name = "Sharing")

# Create a box plot for sharing ratings
ggplot(sharing_long_df, aes(x = Stimulus, y = Sharing)) +
  geom_boxplot() +
  labs(title = "Box Plot of Sharing Ratings", x = "Stimulus", y = "Sharing Ratings")

# Create a violin plot for sharing ratings
ggplot(sharing_long_df, aes(x = Stimulus, y = Sharing)) +
  geom_violin() +
  labs(title = "Violin Plot of Sharing Ratings", x = "Stimulus", y = "Sharing Ratings")

ggplot(sharing_long_df, aes(x = Stimulus, y = Sharing)) +
  geom_violin() +
  stat_summary(fun.data = "median_hilow", geom = "errorbar", width = 0.2) +
  labs(title = "Violin Plot of Sharing Ratings with Mean and Confidence Interval", x = "Stimulus", y = "Sharing Ratings")


# Create a long format data frame for accuracy ratings
accuracy_long_df <- reshape2::melt(clean_dataset, id.vars = NULL, measure.vars = news_accuracy, variable.name = "Stimulus", value.name = "accuracy")

# Create a box plot for sharing ratings
ggplot(accuracy_long_df, aes(x = Stimulus, y = accuracy)) +
  geom_boxplot() +
  labs(title = "Box Plot of accuracy Ratings", x = "Stimulus", y = "accuracy Ratings")

# Create a violin plot for sharing ratings
ggplot(accuracy_long_df, aes(x = Stimulus, y = accuracy)) +
  geom_violin() +
  labs(title = "Violin Plot of accuracy Ratings", x = "Stimulus", y = "accuracy Ratings")

# Create a long format data frame for trust ratings
Trust_long_df <- reshape2::melt(clean_dataset, id.vars = NULL, measure.vars = news_trust, variable.name = "Stimulus", value.name = "trust")

# Create a box plot for trust ratings
ggplot(Trust_long_df, aes(x = Stimulus, y = trust)) +
  geom_boxplot() +
  labs(title = "Box Plot of trust Ratings", x = "Stimulus", y = "trust Ratings")

# Create a violin plot for trust ratings
ggplot(Trust_long_df, aes(x = Stimulus, y = trust)) +
  geom_violin() +
  labs(title = "Violin Plot of trust Ratings", x = "Stimulus", y = "trust Ratings")




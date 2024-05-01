# Load required libraries
setwd("/Users/diptobarman/Desktop/2023/experiment1/USING R/")
library(dplyr)

Dataset <- read.csv("clean_dataset.csv")

df <- Dataset %>%
  select(-c('Personality_Traits_1','Personality_Traits_2', 'Personality_Traits_3','Personality_Traits_4','Personality_Traits_5', 'Personality_Traits_6','Personality_Traits_7','Personality_Traits_8', 'Personality_Traits_9','Personality_Traits_10'))

df <- df %>%
  select(-c(CRT_1,CRT_2,CRT_3,CRT_4,CRT_5,CRT_6,CRT_7,cr_1,cr_2,cr_3,cr_4,cr_5,cr_6,cr_7))

df <- Dataset %>%
  select(c(Gender,Age,Education,Social_media_use,Political_Label,CRT,Fake_news_A,Fake_news_S,Fake_news_Flag_A,Fake_news_Flag_S,Fake_news_Flag_T,True_news_A,True_news_S,
           Fake_news_W_A,Fake_news_W_S,Fake_news_W_T,Openness,Extraversion,Conscientiousness,Neuroticism, Agreeableness))

df$Gender <- as.factor(df$Gender, levels = c(1,2), labels = c('male', 'female','binary', 'prefer not to say'))
df$Age <-as.factor(df$Age, levels = c(1,2,3,4,5,6), labels = c('18-25', '26-35','36-45','46-55','56-65','65+'))
df$Education <- as.factor(df$Education, levels = c(1,2,3,4,5), labels = c('Less than high school', 'high school', 'undergraduate','graduate','postgraduate'))
df$Social_media_use <- as.factor(df$Social_media_use, levels = c(1,2,3,4,5), labels = c('0-1 hour','2-3 hours','4-5 hours','5-6 hours','+6 hours'))
df$Political_Label <- as.factor(df$Political_Label, levels = c(1,2,3,4,5), labels = c('extremely Liberal','moderately Liberal', 'neither liberal nor conservative', 'moderately conservative','extremely conservative'))
df$CRT <- as.factor(df$CRT, levels = c(0,1,2,3,4,5,6,7), labels = c('0','1','2','3','4','5','6','7'))

# List of all the factor variables
factor_vars <- c("Gender", "Age", "Education", "Social_media_use", "Political_Label", "CRT")

library(skimr)
library(gt)
table <- skim(df)

table <- table %>%
  select(-c(skim_type,n_missing,complete_rate,numeric.p0,numeric.p25,numeric.p50,numeric.p75,numeric.p100,numeric.hist))

names(table) <- c("Variable", "Mean", "standard deviation")

table <- table[-c(1,2,3,4,5), ]

# Create a gt table
gt_table <- gt(table)

# Display the table
print(gt_table)

# Create a bar plot for each variable
for (var in factor_vars) {
  print(
    ggplot(Dataset, aes_string(x = var)) +
      geom_bar() +
      xlab(var) +
      ylab("Count") +
      ggtitle(paste("Distribution of", var))
  )
}

# Reshape the data to a long format
long_data <- df %>%
  gather(key = "Variable", value = "Value", Gender, Age, Education, Social_media_use, Political_Label, CRT)

# Reorder factor levels
long_data$Value <- factor(long_data$Value, levels = c('male', 'female', 'binary', 'prefer not to say', 
                                                      '18-25', '26-35', '36-45', '46-55', '56-65', '65+', 
                                                      'Less than high school', 'high school', 'undergraduate', 'graduate', 'postgraduate', 
                                                      '0-1 hour', '2-3 hours', '4-5 hours', '5-6 hours', '+6 hours', 
                                                      'extremely Liberal', 'moderately Liberal', 'neither liberal nor conservative', 'moderately conservative', 'extremely conservative',
                                                      '0','1','2','3','4','5','6','7'))

# Create the plot
ggplot(long_data, aes(x=Value)) +
  geom_bar() +
  facet_wrap(~ Variable, scales = "free") +
  xlab("Variable") +
  ylab("Count") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))


# Select numeric variables only
numeric_data <- Dataset[, c("Age", "Education", "Social_media_use", "Political_Label", "CRT", 
                                                         "Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism",
                                                        "Fake_news_A", "True_news_A", "Fake_news_Flag_A",
                                                        "Fake_news_W_A", "Fake_news_S", "True_news_S",
                                                       "Fake_news_Flag_S", "Fake_news_W_S",
                                                       "Fake_news_Flag_T", "Fake_news_W_T")]

# Convert factors to numeric levels, if they are ordered factors
numeric_data$Age <- as.numeric(as.character(numeric_data$Age))
numeric_data$Education <- as.numeric(as.character(numeric_data$Education))
numeric_data$Social_media_use <- as.numeric(as.character(numeric_data$Social_media_use))
numeric_data$Political_Label <- as.numeric(as.character(numeric_data$Political_Label))




# Install and load the corrplot package
if (!require(corrplot)) {
  install.packages("corrplot")
}
library(corrplot)

# Create the correlation heatmap
corrplot(cor_matrix, method = "color", type = "upper", 
         title = "Correlation Heatmap", mar = c(0,0,1,0), 
         tl.col = "black", tl.srt = 45)


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

----- ### Something New to see

  
df$Gender <- factor(df$Gender, levels = c(1, 2, 3, 4), labels = c('male', 'female', 'binary', 'prefer not to say'))
df$Age <- factor(df$Age, levels = c(1, 2, 3, 4, 5, 6), labels = c('18-25', '26-35', '36-45', '46-55', '56-65', '65+'))
df$Education <- factor(df$Education, levels = c(1, 2, 3, 4, 5), labels = c('Less than high school', 'high school', 'undergraduate', 'graduate', 'postgraduate'))
df$Social_media_use <- factor(df$Social_media_use, levels = c(1, 2, 3, 4, 5), labels = c('0-1 hour', '2-3 hours', '4-5 hours', '5-6 hours', '+6 hours'))
df$Political_Label <- factor(df$Political_Label, levels = c(1, 2, 3, 4, 5), labels = c('extremely Liberal', 'moderately Liberal', 'neither liberal nor conservative', 'moderately conservative', 'extremely conservative'))
df$CRT <- factor(df$CRT, levels = c(0, 1, 2, 3, 4, 5, 6, 7), labels = c('0', '1', '2', '3', '4', '5', '6', '7'))



# Kruskal-Wallis test for Gender and Fake_news_Flag_T
kruskal.test(Fake_news_Flag_T ~ Education, data = df)

# Kruskal-Wallis test for Gender and Fake_news_W_T
kruskal.test(Fake_news_W_T ~ Education, data = df)

install.packages("lme4")
install.packages("Matrix")
library(Matrix)
library(lme4)

# Assuming df is your dataframe
df$ParticipantID <- 1:nrow(df)


model <- glmer(Fake_news_W_T ~ Gender + Age + Education + Social_media_use + Political_Label + CRT + (1|ParticipantID), 
               family = Gamma(link = "log"), 
               data = df, 
               control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 100000)))

summary(model)





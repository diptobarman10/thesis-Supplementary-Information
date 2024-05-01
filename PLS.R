# Independent variables
independent_vars <- c("Gender", "Education", "Social_media_use", "Political_Label", "Age", "CRT", "Openness", "Extraversion", "Neuroticism", "Agreeableness", "Conscientiousness")

# Dependent variables
dependent_vars <- c("Fake_news_A", "Fake_news_Flag_A", "Fake_news_W_A", "True_news_A",
                    "Fake_news_S", "Fake_news_Flag_S", "Fake_news_W_S", "True_news_S",
                    "Fake_news_Flag_T", "Fake_news_W_T")

# 
lm = lm(Fake_news_Flag_T ~ Openness + Extraversion + Neuroticism 
        + Agreeableness + Conscientiousness + CRT + Age + Political_Label + Social_media_use + Education, 
         data = df)

summary(lm)

install.packages("lme4")
install.packages("Matrix")
library(Matrix)
library(lme4)

glm_model <- glm(Fake_news_W_T ~ + Neuroticism + Agreeableness + CRT  + 
                   Political_Label + Social_media_use , data = df)
summary(glm_model)

install.packages("pls")
install.packages("plspm")
library(plspm)
library(pls)
plsm = plsr(Fake_news_W_T ~ Openness + Extraversion + Neuroticism 
        + Agreeableness + Conscientiousness + CRT + Age + Political_Label + Social_media_use + Education + Gender
        , data = df, ncomp=11, validation="CV")

summary(plsm)

plot(RMSEP(plsm))

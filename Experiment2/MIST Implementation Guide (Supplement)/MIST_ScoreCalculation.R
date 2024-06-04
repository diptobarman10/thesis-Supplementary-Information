###########################
## MIST - Score Calculation
###########################

colnames(dat) <- sub("MIST20", "MIST", colnames(dat))

##
## Select Columns
##

## MIST-8
MIST8_Columns <- c("MIST_1", "MIST_3", "MIST_6", "MIST_8",
                   "MIST_11", "MIST_15", "MIST_17", "MIST_19")
MIST8_Columns_Fake <- c("MIST_1", "MIST_3", "MIST_6", "MIST_8")
MIST8_Columns_Real <- c("MIST_11", "MIST_15", "MIST_17", "MIST_19")

## MIST-20
MIST20_Columns <- sprintf("MIST_%d", 1:20)
MIST20_Columns_Fake <- sprintf("MIST_%d", 1:10)
MIST20_Columns_Real <- sprintf("MIST_%d", 11:20)

##
## Convert Values
##

dat[,MIST20_Columns] <- sapply(dat[,MIST20_Columns],as.character)
dat[,MIST20_Columns_Fake][dat[,MIST20_Columns_Fake] == "Fake"] <- "1"
dat[,MIST20_Columns_Fake][dat[,MIST20_Columns_Fake] == "Real"] <- "0"
dat[,MIST20_Columns_Real][dat[,MIST20_Columns_Real] == "Fake"] <- "0"
dat[,MIST20_Columns_Real][dat[,MIST20_Columns_Real] == "Real"] <- "1"
dat[,MIST20_Columns] <- sapply(dat[,MIST20_Columns],as.numeric)

##
## Calculate Scores
##

## MIST-20
# V | Veracity Discernment
dat$MIST20_V <- rowSums(dat[,MIST20_Columns])

# r | Real News Detection Ability
dat$MIST20_r <- rowSums(dat[,MIST20_Columns_Real])

# f | Fake News Detection Ability
dat$MIST20_f <- rowSums(dat[,MIST20_Columns_Fake])

# d | Distrust (Negative Response Bias)
dat$MIST20_d <- rowSums(abs(1-dat[,MIST20_Columns_Real]))+rowSums(dat[,MIST20_Columns_Fake])-10
dat$MIST20_d[which(dat$MIST20_d < 0)] <- 0

# n | Naïvité (Positive Response Bias)
dat$MIST20_n <- rowSums(abs(1-dat[,MIST20_Columns_Fake]))+rowSums(dat[,MIST20_Columns_Real])-10
dat$MIST20_n[which(dat$MIST20_n < 0)] <- 0

# Category
dat$MIST20_Category <- dat$MIST20_V
dat$MIST20_Category[dat$MIST20_Category > summary(dat$MIST20_V)[3]] <- "High"
dat$MIST20_Category[dat$MIST20_Category != "High"] <- "Low"
dat$MIST20_Category <- factor(dat$MIST20_Category, levels = c("Low", "High"))

## MIST-8
# V | Veracity Discernment
dat$MIST8_V <- rowSums(dat[,MIST8_Columns])

# r | Real News Detection Ability
dat$MIST8_r <- rowSums(dat[,MIST8_Columns_Real])

# f | Fake News Detection Ability
dat$MIST8_f <- rowSums(dat[,MIST8_Columns_Fake])

# d | Distrust (Negative Response Bias)
dat$MIST8_d <- rowSums(abs(1-dat[,MIST8_Columns_Real]))+rowSums(dat[,MIST8_Columns_Fake])-4
dat$MIST8_d[which(dat$MIST8_d < 0)] <- 0

# n | Naïvité (Positive Response Bias)
dat$MIST8_n <- rowSums(abs(1-dat[,MIST8_Columns_Fake]))+rowSums(dat[,MIST8_Columns_Real])-4
dat$MIST8_n[which(dat$MIST8_n < 0)] <- 0

# Category
dat$MIST8_Category <- dat$MIST8_V
dat$MIST8_Category[dat$MIST8_Category > summary(dat$MIST8_V)[3]] <- "High"
dat$MIST8_Category[dat$MIST8_Category != "High"] <- "Low"
dat$MIST8_Category <- factor(dat$MIST8_Category, levels = c("Low", "High"))

###Dataracing 2023 overview The Dataracing 2023 competition focuses on domestic lending data and defaults, looking at how predictable it is for a borrower to default on a loan. For this task, the anonymised and distorted data is provided by the co-organiser of the competition, the Hungarian National Bank. The aim of the data analysis is to predict as accurately as possible whether a customer is likely to default on a loan in the two years following the period covered by the data. https://eval.dataracing.hu/web/challenges/challenge-page/5/overview

#Link to the dataset: https://drive.google.com/drive/folders/1xPFJ_Ln0lo12oJZk7GN03e_vvCl8R6m9
#Submitted solutions (.csv files) are evaluated by the Dataracing platform. The platform uses the log loss (sklearn.metric.log_loss) metrics. Accordingly, we examine how well the prediction submitted by the competitors corresponds to a good probability value when the value of the objective function is considered.
# I prepared for the task a simple logistic regression model, with WOE binning

install.packages('gmodels')
install.packages("ROCR")
install.packages("pROC")
install.packages("tables")
install.packages("caret")
install.packages("C50") 
install.packages('psych')
install.packages("VIM")
install.packages("scorecard")
install.packages("dplyr")


library(gmodels)
library(ROCR)
library (pROC)
library(plyr)
library(caret)
library(C50)
library('psych')
library(scorecard)
library(VIM)
library(dplyr)

#Read the data and first steps for data handling
adatok_dr <- read.csv("training_data.csv", na.strings=c(""," ", "NA", "n/a"))

adatok_dr <- as.data.frame(adatok_dr) 
output <- read.csv('data_submission_example.csv')
output <- as-data.frame(output)

##DATA TRANSFORMATIONS
#Dates as new column
adatok_dr$CONTRACT_DATE_OF_LOAN_AGREEMENT2 <- as.Date(adatok_dr$CONTRACT_DATE_OF_LOAN_AGREEMENT, origin= "1970-01-01")
adatok_dr$CONTRACT_MATURITY_DATE2 <- as.Date(adatok_dr$CONTRACT_MATURITY_DATE, origin = "1970-01-01")
adatok_dr$TARGET_EVENT_DAY2 <- as.Date(adatok_dr$TARGET_EVENT_DAY, origin = "1970-01-01")

#time after loan disbursement as new column 
adatok_dr$PAST_TIME <- as.numeric(adatok_dr$TARGET_EVENT_DAY2  - adatok_dr$CONTRACT_DATE_OF_LOAN_AGREEMENT2)

#Set the target variable (deafult loans)
adatok_dr$TARGET_EVENT
adatok_dr$TARGET_EVENT2 <- ifelse (adatok_dr$TARGET_EVENT2 == 'K' & adatok_dr$PAST_TIME < 730, 0, ifelse (adatok_dr$TARGET_EVENT2=='E' & adatok_dr$PAST_TIME < 730,'E', '1'))

#We need only the cases where we have minimum 2 years credit history or it has been already indefault (dropping the other rows)
adatok_dr2 <- adatok_dr[adatok_dr$TARGET_EVENT2 != 'E', ]
summary(adatok_dr2)

table(adatok_dr2$TARGET_EVENT2)
adatok_dr2$TARGET_EVENT2 <- as.numeric(adatok_dr2$TARGET_EVENT2)

#The breakdown of the cases by product type:
# 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 67384 991021 57995 90581 21000 2247 154013 34608 78376 51518 6771 3350 103 2654 569 70 240 3

#Product 1,2, 3, 7, 9, 10 are relevant. We investigate and build separate models for those prducts.

####Create a table for WOE binning and change type of the factors

adatok_woe_o <- adatok_dr

adatok_woe_o$CONTRACT_CREDIT_INTERMEDIARY <- as.factor(adatok_woe_o$CONTRACT_CREDIT_INTERMEDIARY)
adatok_woe_o$BORROWER_TYPE_OF_CUSTOMER <- as.factor(adatok_woe_o$BORROWER_TYPE_OF_CUSTOMER)
adatok_woe_o$CONTRACT_FREQUENCY_TYPE <- as.factor(adatok_woe_o$CONTRACT_FREQUENCY_TYPE)
adatok_woe_o$CONTRACT_BANK_ID <- as.factor(adatok_woe_o$CONTRACT_BANK_ID)
adatok_woe_o$CONTRACT_LOAN_CONTRACT_TYPE <- as.factor(adatok_woe_o$CONTRACT_LOAN_CONTRACT_TYPE)
adatok_woe_o$CONTRACT_LOAN_TYPE <- as.factor(adatok_woe_o$CONTRACT_LOAN_TYPE)
adatok_woe_o$CONTRACT_LOAN_AMOUNT <- as.numeric(adatok_woe_o$CONTRACT_LOAN_AMOUNT)
adatok_woe_o2$CONTRACT_CURRENCY <- as.factor(adatok_woe_o$CONTRACT_CURRENCY)
adatok_woe_o$TARGET_EVENT2 <- as.factor(adatok_woe_o$TARGET_EVENT2)

#WOE binning for one product (7)
adatok_woe_7 <- subset(adatok_woe_o, CONTRACT_LOAN_TYPE ==7)
woe_er7 <- woebin(adatok_woe_7, y = 'TARGET_EVENT2', x =c("CONTRACT_BANK_ID", "CONTRACT_CREDIT_INTERMEDIARY", "CONTRACT_CREDIT_LOSS", "CONTRACT_CURRENCY", "CONTRACT_FREQUENCY_TYPE", "CONTRACT_INCOME", "CONTRACT_INSTALMENT_AMOUNT", "CONTRACT_INSTALMENT_AMOUNT_2", "CONTRACT_INTEREST_PERIOD", "CONTRACT_INTEREST_RATE", "CONTRACT_LGD", "CONTRACT_LOAN_AMOUNT", "CONTRACT_LOAN_CONTRACT_TYPE", "CONTRACT_LOAN_TO_VALUE_RATIO", "CONTRACT_LOAN_TYPE","CONTRACT_MARKET_VALUE",  "CONTRACT_MORTGAGE_LENDING_VALUE","CONTRACT_MORTGAGE_TYPE", "CONTRACT_REFINANCED", "CONTRACT_RISK_WEIGHTED_ASSETS", "CONTRACT_TYPE_OF_INTEREST_REPAYMENT", "BORROWER_BIRTH_YEAR", "BORROWER_CITIZENSHIP", "BORROWER_COUNTRY", "BORROWER_COUNTY", "BORROWER_TYPE_OF_CUSTOMER", "BORROWER_TYPE_OF_SETTLEMENT"))

adatok_dr_t7_woe <- woebin_ply(adatok_dr_t7, woe_er7)

#check the bins and the information values
summary(adatok_dr_t7_woe)
woebin_plot(woe_er7, x = NULL, title = NULL, show_iv = TRUE)

#Logistic regression model building
adatok_dr_t7_woe$TARGET_EVENT2 <- as.numeric(adatok_dr_t7_woe$TARGET_EVENT2)

adatok_lreg_t7 <- adatok_dr_t7_woe[,c(1:2, 13:39)] 

set.seed(2000)
g<- runif(nrow(adatok_lreg_t7))
adatok_l7 <- adatok_lreg_t7[order(g),]

train7 <- adatok_l7[1:120000, c(5:30)]
test7 <- adatok_l7[120001:nrow(adatok_l7) , 6:30] 

test_teljes7 <- adatok_l7[120001:nrow(adatok_l7), ]
lreg7 <- glm(TARGET_EVENT2~., family = binomial, data = train7)

log_predict7 <- predict(lreg7, type = "response", newdata = test7[,])
pred_risk_l7 <- factor(ifelse(log_predict7 >= 0.999 , "Paid", "Default")) #finetuning of the cutting value if necessary, based on ROCIT
actual_risk7 <- factor(ifelse(test_teljes7$TARGET_EVENT2=='1',"Paid", "Default"))

#Evaulation
table(actual_risk7,pred_risk_l7)
CrossTable(actual_risk7, pred_risk_l7)

pred_risk_lreg7 <- as.numeric(log_predict7)
ROCit_obj7 <- rocit(score=pred_risk_lreg7,class=actual_risk7)
plot(ROCit_obj7)
summary(ROCit_obj7)
confusionMatrix(reference=actual_risk7, data=pred_risk_l7, mode = "everything")

#Preparing output table as requested
adatok_lreg_t7$prediction <- predict(lreg7, type = "response", newdata = adatok_lreg_t7[,])
adatok_lreg_t7$PD <- 1-adatok_lreg_t7$prediction
nrow(adatok_lreg_t7)
head(adatok_lreg_t7)

#### The same process for all product, but after cheking the information value of the variables, drop the unnecessary variables, customizing the model, and setting the best possible cutoff value. Finally concat the total output table.
merged_PD <- rbind(adatok_lreg_t1, adatok_lreg_t10, adatok_lreg_t2, adatok_lreg_t3, adatok_lreg_t4, adatok_lreg_t7, adatok_lreg_t9, adatok_lreg_t99, fill=TRUE)

merged_2 <- merged_PD[ , c('BORROWER_ID', 'PD')]
merged_2 <- merged_2[!duplicated(merged_2$BORROWER_ID), ]




###replacing NA with mean 
merged_2$PD[is.na(merged_2$PD)] <- mean(merged_2$PD, na.rm = TRUE) 

output_veg <- merge(output, merged_2, by='BORROWER_ID', all.x = TRUE, all.y = FALSE)

#### inputing missing values and drop predicted values 
output_veg$PD[is.na(output_veg$PD)] <- mean(output_veg$PD, na.rm = TRUE) 
output_veg <- output_veg[ , c('BORROWER_ID', 'PD')]
colnames(output_veg)[colnames(output_veg) == "PD"] <- "PRED"

############ write csv
write.csv(output_veg, file = 'merged2.csv', row.names = FALSE)

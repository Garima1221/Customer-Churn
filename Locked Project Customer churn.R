## Clear R environment
rm(list = ls())

## Load libraries
lib_set = c("ggplot2","corrplot","e1071","usdm","caret","class","randomForest","C50")
lapply(lib_set,require,character.only = TRUE)

#Set working directory
setwd("C:/Users/Garima/Downloads/Edwisor/Project Customer Churn")

train_set = read.csv("Train_data.csv",header = TRUE,na.strings = c(" ",""))
test_set = read.csv("Test_data.csv",header = TRUE,na.strings = c(" ",""))

## combine train and test set 
##1:3333 train set
##3334:5000 test set 
data_set = rbind(train_set,test_set)

## a brief gist of data set 
summary(train_set)
##483/3333 ~ 14 % churn rate in the train set provided
summary(test_set)


## Let us analyse the feature set provided 
colnames(data_set)
str(data_set)

#Convert some string variables to numeric factors 
data_set$area.code = as.factor(data_set$area.code)
data_set$Churn = as.factor(ifelse(data_set$Churn == " True.",1,0))
data_set$international.plan = as.factor(ifelse(data_set$international.plan == " yes",1,0))
data_set$voice.mail.plan = as.factor(ifelse(data_set$voice.mail.plan == " yes",1,0))

##convert factor variables to levels
for(i in c("state","area.code","phone.number")){
  if(class(data_set[,i]) == 'factor'){
    print(data_set["area.code"])
    data_set[,i] = factor(data_set[,i],labels = 1:length(levels(factor(data_set[,i]))))
    print(data_set["area.code"])
  }
}

############### Missing Values Analysis############################
## Data does not have any missing values
sum(is.na(data_set))


########## Feature Engineering ###################################
#colnames(data_set)
## Churn rate is higher for higher calls and higher rates
##data_set$total_charge = data_set$total.day.charge+data_set$total.eve.charge+data_set$total.night.charge+data_set$total.intl.charge
#data_set$total_calls = data_set$total.day.calls+data_set$total.eve.calls+data_set$total.night.calls+data_set$total.intl.calls

## per minute rates are low ,still churn is higher
#data_set$day_rate = data_set$total.day.charge/data_set$total.day.minutes
#data_set$eve_rate  = data_set$total.eve.charge/data_set$total.eve.minutes
#data_set$night_rate = data_set$total.night.charge/data_set$total.night.minutes
#data_set$intl_rate = data_set$total.intl.charge/data_set$total.intl.minutes

############################## Visualisations #####################################
## We analyse the categorical feature wise Churn rate for train set 
train_set = data_set[1:nrow(train_set),]
feature_wise_churn_percent = function(feature_name,target_class){
  churn_table = table(feature_name,target_class)
  churn_percent = (churn_table[,2]/(churn_table[,1]+churn_table[,2]))*100
  return(sort(churn_percent,decreasing = TRUE))
}

for(feature in c("state","area.code","international.plan","voice.mail.plan")){
  print(feature)
  print(feature_wise_churn_percent(train_set[,feature],train_set[,"Churn"]))
}


dev.off()

for(i in c("account.length","number.vmail.messages","total.day.calls","total.day.charge","total.eve.calls","total.eve.charge","total.night.calls","total.night.charge","total.intl.calls","total.intl.charge","number.customer.service.calls")){
  #dev.off()
  print(i)
  hist(train_set[,i],main = i,xlab=i)
  assign(paste0("gg_",i),ggplot(aes_string(x = "Churn",y = i),data = subset(train_set),group = 2)
         + stat_boxplot(geom = "errorbar",width = 0.3) +
           geom_boxplot(outlier.colour = "red",fill = "blue",outlier.shape = 18,outlier.size = 1) +
           labs(y=i,x = "Churn") + 
           ggtitle(i))
}

gridExtra::grid.arrange(gg_account.length,gg_number.vmail.messages,nrow = 2,ncol=1)
gridExtra::grid.arrange(gg_total.day.calls,gg_total.day.charge,nrow = 1,ncol = 2)
gridExtra::grid.arrange(gg_total.eve.calls,gg_total.eve.charge,nrow = 2,ncol = 1)
gridExtra::grid.arrange(gg_total.night.calls,gg_total.night.charge,nrow = 2,ncol = 1)

######################################Outlier Analysis #####################################
for(i in c("account.length","number.vmail.messages","total.day.minutes","total.day.calls","total.day.charge","total.eve.minutes","total.eve.calls","total.eve.charge","total.night.minutes","total.night.calls","total.night.charge","total.intl.minutes","total.intl.calls","total.intl.charge","number.customer.service.calls")){
  outlier_value = boxplot.stats(data_set[,i])$out
  print(i)
  print(outlier_value)
  data_set[which(data_set[,i] %in% outlier_value),i] = NA
  data_set[is.na(data_set[,i]),i] = round(mean(data_set[,i],na.rm = TRUE))
  
}
###################################### Feature Selection ####################################
##numerical feature
## Correlation check 

correlation_matrix = cor(data_set[,c("account.length","number.vmail.messages","total.day.minutes","total.day.calls","total.day.charge","total.eve.minutes","total.eve.calls","total.eve.charge","total.night.minutes","total.night.calls","total.night.charge","total.intl.minutes","total.intl.calls","total.intl.charge","number.customer.service.calls")])
corrplot(correlation_matrix,method = "number",type = "lower")
#delete features having high correlation

#drop total.night.minutes,total.intl.minutes,total.eve.minutes,total.day.minutes

##Multicollinearity

vif(data_set[,c("account.length","number.vmail.messages","total.day.minutes","total.day.calls","total.day.charge","total.eve.minutes","total.eve.calls","total.eve.charge","total.night.minutes","total.night.calls","total.night.charge","total.intl.minutes","total.intl.calls","total.intl.charge","number.customer.service.calls")])


## categorical data
for(i in c("state","area.code","phone.number","international.plan","voice.mail.plan")){
  print(i)
  print(chisq.test(table(data_set$Churn,data_set[,i]),simulate.p.value = TRUE))
}
##phone number ,areacode
data_set = subset(data_set,select = -c(phone.number,area.code,total.night.minutes,total.intl.minutes,total.eve.minutes,total.day.minutes))


#### Let us oversample the minority class and undersample the majority class

library(ROSE)
data_balanced_both = ovun.sample(Churn~.,train_set, method = "both",p = 0.5)$data
table(data_balanced_both$Churn)

########## Model Development #########################################
## separate train and test set in the data 
train_set = data_set[1:nrow(train_set),]
x_train = subset(train_set,select = -c(Churn))
y_train = train_set$Churn
test_set = data_set[(nrow(train_set)+1):nrow(data_set),]
x_test = subset(test_set,select = -c(Churn))
y_test = test_set$Churn

###################### Model Development #########################################
####### Logistic Regression
set.seed(11122018)

model.LR = glm(Churn~.,data = train_set,family = "binomial")
summary(model.LR)
predict.LR = predict(model.LR,x_test,type = "response")
y.pred = ifelse(predict.LR>0.50,1,0)
confusion_matrix = table(y_test,y.pred)
#accuracy = 87.34%
#FNR = 84.83%


## Naive Bayes 
model.NB = naiveBayes(Churn~.,data = train_set)
predict.NB = predict(model.NB,x_test,type = "class")
confusion_matrix = table(y_test,predict.NB)
##Accuracy = 87.94%
## FNR = 77.69%

## KNN
model.knn = knn(x_train,x_test,y_train,k=3)
confusion_matrix = table(y_test,model.knn)
#Accuracy = 84.94%
##FNR = 88.39%

## Decision Trees

model.DT = C5.0(Churn~.,train_set,trials = 100,rules = TRUE)
summary(model.DT)
predict.DT = predict(model.DT,x_test,type = "class")
confusion_matrix = table(y_test,predict.DT)
#Accuracy = 92.80
#FNR = 52.23

## Random Forest

## out of bag error
model.RF = randomForest(Churn~.,data = train_set,importance = TRUE,ntree = 1000)
importance(model.RF)
varImpPlot(model.RF)
predict.RF = predict(model.RF,x_test)
confusion_matrix = table(y_test,predict.RF)
#Accuracy = 100
#FNR = 0



########### Example of output with sample input 
#### We see randm forest fairs well on our given data set,thus we use this model 
#### Let us take sample input from train data and produce output using random forest 
sample_index = createDataPartition(test_set$Churn,p = 0.50,times = 1,list = FALSE)
sample_data = test_set[sample_index,]
predict.RF = predict(model.RF,sample_data[,-15])
confusion_matrix = table(sample_data$Churn,predict.RF)
## Accuracy :100
## FNR :0

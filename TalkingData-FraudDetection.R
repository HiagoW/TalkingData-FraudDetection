setwd("C:/Users/hiago/OneDrive/Cursos/FCD/BigDataRAzure/Projetos-1-2/Projeto01/")

# Lendo arquivo
df <- read.csv('train_sample.csv')
head(df)
str(df)
View(df)

# Convertendo variável target para fator
df$is_attributed <- as.factor(df$is_attributed)

# Análise exploratória
library(ggplot2)
library(dplyr)

# Proporção entre as classes
ggplot(df,aes(x=is_attributed)) +
  geom_bar()

table(df$is_attributed)

table(df$is_attributed)[1]/nrow(df)
table(df$is_attributed)[2]/nrow(df)

# Classes bem desbalanceadas, muito mais fraudes
# Balanceando classes com SMOTE
# Utilizei o módulo do Azure Machine Learning para balancear as classes, depois fiz o download do csv

smoted_df <- read.csv('smoted_df.csv')

# Undersampling - Reduzindo número de classes = 0
smoted_df_sample <- smoted_df %>% filter(is_attributed == 0) %>% 
  sample_n(3000)

# Juntando o resultado do undersampling com as linhas onde o target = 1
smoted_df_sample <- rbind(smoted_df_sample,smoted_df[smoted_df$is_attributed==1,])

table(smoted_df_sample$is_attributed)

smoted_df_sample$is_attributed = as.factor(smoted_df_sample$is_attributed)

# Dividindo em dados de treino e test
library(caret)
library(randomForest)

amostra <- createDataPartition(smoted_df_sample$is_attributed,p=0.7,list=F)
testData <- smoted_df_sample[amostra,]
trainData <- smoted_df_sample[-amostra,]

# Verificando proporção das classes
table(testData$is_attributed)
table(trainData$is_attributed)

df_rf <- smoted_df_sample

# Removendo variáveis de tempo, irrelevantes para a construção do modelo pois apenas indicam quando a operação foi feita
df_rf$attributed_time = NULL
df_rf$click_time = NULL

str(df_rf)

# Verificando se tem valores NA
any(is.na(df_rf))

?randomForest

# ------------------------ Modelo V1 Random Forest --------------------------------
modelo_rf_v1 <- randomForest(is_attributed ~ .,data=df_rf)

summary(modelo_rf_v1)

predictions <- predict(modelo_rf_v1, testData)

# 99.79 %
mean(predictions == testData$is_attributed)

# AUC
library(pROC)
library(ROCR)
roc_obj <- roc(testData$is_attributed, as.numeric(predictions))
roc_obj

# Função para plot
rocplot <- function(pred, truth, ...) {
  predob = prediction(pred, truth)
  perf = performance(predob, "tpr", "fpr")
  plot(perf, ...)
  area <- auc(truth, pred)
  area <- format(round(area, 4), nsmall = 4)
  text(x=0.8, y=0.1, labels = paste("AUC =", area))
  
  # the reference x=y line
  segments(x0=0, y0=0, x1=1, y1=1, col="gray", lty=2)
}

rocplot(as.numeric(predictions),testData$is_attributed, col="blue")

# Importância das variáveis
var_imp <- randomForest(is_attributed ~ .,data=df_rf,importance=T)
varImpPlot(var_imp)

# Removendo a coluna OS
df_rf2 = df_rf

df_rf2$os = NULL

# -------------------------- Modelo V2 Random Forest -------------------------------
modelo_rf_v2 <- randomForest(is_attributed ~ .,data=df_rf2)

summary(modelo_rf_v2)

predictions <- predict(modelo_rf_v2, testData)

# 99.45 % - Piorou
mean(predictions == testData$is_attributed)

#AUC
roc_obj <- roc(testData$is_attributed, as.numeric(predictions))
roc_obj

rocplot(as.numeric(predictions),testData$is_attributed, col="blue")

# ---------------------------- SVM -----------------------------------
library(e1071)

modelo_svm_v1 = svm(is_attributed ~ .,data=df_rf)

summary(modelo_svm_v1)

predictions <- predict(modelo_svm_v1, testData)

# 86.64 % - Piorou
mean(predictions == testData$is_attributed)

#AUC
roc_obj <- roc(testData$is_attributed, as.numeric(predictions))
roc_obj

rocplot(as.numeric(predictions),testData$is_attributed, col="blue")

# Modelo final = modelo_rf_v1


# -------------------------------- Predicts do csv do Kaggle ------------------------
# library(readr)

# df_test <- read_csv("C:/Users/hiago/Downloads/test.csv")
# df_preds = data.frame(click_id=df_test$click_id)

# df_test$click_time = NULL
# df_test$click_id = NULL

# preds <- predict(modelo_rf_v1,df_test)

# table(preds)

# df_preds$is_attributed = preds

# table(df_preds$is_attributed)

# View(df_preds)

#install.packages("csvread")
# library(csvread)

# df_preds$click_id = as.int64(df_preds$click_id)

# ?write.table
# write.table(df_preds,"C:/Users/hiago/Downloads/submission.csv",sep=",",row.names = FALSE)



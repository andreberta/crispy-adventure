library(xgboost)

doing <- "dog"

if (doing == "cat") {
    train <- read.csv("./clean_data/cat_train.csv")
    Xs_test <- read.csv("./clean_data/cat_test.csv")
    out_file <- "./predictions/XGB_cat.csv"
    n_rounds <- 20
} else {
    train <- read.csv("./clean_data/dog_train.csv")
    Xs_test <- read.csv("./clean_data/dog_test.csv")
    out_file <- "./predictions/XGB_dog.csv"
    n_rounds <- 30
}

#as attributes we have
#attributes = [:AnimalType, :AgeinDays, :HasName, :NameLength, :Hour, :Minute, :Weekday, :Day, :Month, :Year, :TimeofDay, :ColorComplexity, :BreedComplexity, :IsMix, :IsSlash :SimpleBreed, :SimpleColor, :Intact, :Sex, :LifeStage]


factorVars <- c('HasName', 'TimeofDay', 'IsMix', 'IsSlash', 'SimpleBreed','SimpleColor','Intact','Sex','LifeStage', 'OutcomeType')
train[factorVars] <- lapply(train[factorVars], function(x) as.factor(x))
factorVars <- c('HasName', 'TimeofDay', 'IsMix', 'IsSlash', 'SimpleBreed','SimpleColor','Intact','Sex','LifeStage')
Xs_test[factorVars] <- lapply(Xs_test[factorVars], function(x) as.factor(x))

# Need to change to numeric
y_train <- as.numeric(as.factor(train$OutcomeType)) - 1

# keep track of the labels
labels_train <- data.frame(train$OutcomeType, y_train)

# xgboost-specific design matrices
xgb_train <- xgb.DMatrix(model.matrix(~AgeinDays+HasName+NameLength+Hour+Minute+Weekday+Day+Month+Year+TimeofDay+ColorComplexity+BreedComplexity+IsMix+IsSlash+SimpleBreed+SimpleColor+Intact+Sex+LifeStage, data=train), label=y_train, missing=NA)

xgb_test  <- xgb.DMatrix(model.matrix(~AgeinDays+HasName+NameLength+Minute+Weekday+Day+Month+Year+TimeofDay+ColorComplexity+BreedComplexity+IsMix+IsSlash+SimpleBreed+SimpleColor+Intact+Sex+LifeStage, data=Xs_test), missing=NA)


# tune nrounds - uncomment first
#xgb.cv(data=xgb_train, label=y_train, nfold=5, nround=200, objective='multi:softprob', num_class=5, eval_metric='mlogloss')

# looks like nrounds should be around 45 (40 gatti, 30 cani per me)


# build model
xgb_model <- xgboost(xgb_train, y_train, nrounds=n_rounds, objective='multi:softprob', num_class=5, eval_metric='mlogloss', early.stop.round=TRUE)

# make predictions
predictions <- predict(xgb_model, xgb_test)

# reshape predictions
xgb_preds <- data.frame(t(matrix(predictions, nrow=5, ncol=length(predictions)/5)))

# name columns
colnames(xgb_preds) <- c('Adoption', 'Died', 'Euthanasia', 'Return_to_owner', 'Transfer')

# attach ID column
xgb_preds['ID'] <- Xs_test['ID']

write.csv(xgb_preds, out_file, row.names=FALSE)


#visualize feature importance
importance_matrix <- xgb.importance(c('AgeinDays', 'HasName', 'NameLength', 'Hour', 'Minute', 'Weekday', 'Day', 'Month', 'Year', 'TimeofDay', 'ColorComplexity', 'BreedComplexity', 'IsMix', 'IsSlash', 'SimpleBreed', 'SimpleColor', 'Intact', 'Sex', 'LifeStage'), model = xgb_model)
xgb.plot.importance(importance_matrix)

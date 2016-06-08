library(xgboost)

train <- read.csv("./clean_data/train.csv")
Xs_test <- read.csv("./clean_data/Xs_test.csv")

factorVars <- c('AnimalType','SimpleBreed','SimpleColor', 'HasName','IsMix','Intact','Sex','TimeofDay','LifeStage', 'OutcomeType')
train[factorVars] <- lapply(train[factorVars], function(x) as.factor(x))
factorVars <- c('AnimalType','SimpleBreed','SimpleColor', 'HasName','IsMix','Intact','Sex','TimeofDay','LifeStage')
Xs_test[factorVars] <- lapply(Xs_test[factorVars], function(x) as.factor(x))

# Need to change to numeric
y_train <- as.numeric(as.factor(train$OutcomeType)) - 1

# keep track of the labels
labels_train <- data.frame(train$OutcomeType, y_train)

# xgboost-specific design matrices
xgb_train <- xgb.DMatrix(model.matrix(~AnimalType+AgeinDays+HasName+Hour+Weekday+Day+Month+Year+TimeofDay+ColorComplexity+BreedComplexity+IsMix+SimpleBreed+SimpleColor+Intact+Sex+LifeStage, data=train), label=y_train, missing=NA)

xgb_test <- xgb.DMatrix(model.matrix(~AnimalType+AgeinDays+HasName+Hour+Weekday+Day+Month+Year+TimeofDay+ColorComplexity+BreedComplexity+IsMix+SimpleBreed+SimpleColor+Intact+Sex+LifeStage, data=Xs_test), missing=NA)


# tune nrounds - uncomment first
#xgb.cv(data=xgb_train, label=y_train, nfold=5, nround=200, objective='multi:softprob', num_class=5, eval_metric='mlogloss')

# looks like nrounds should be around 45 (mah, per me gia a 20 è ok...)


# build model
xgb_model <- xgboost(xgb_train, y_train, nrounds=40, objective='multi:softprob', num_class=5, eval_metric='mlogloss', early.stop.round=TRUE)

# make predictions
predictions <- predict(xgb_model, xgb_test)

# reshape predictions
xgb_preds <- data.frame(t(matrix(predictions, nrow=5, ncol=length(predictions)/5)))

# name columns
colnames(xgb_preds) <- c('Adoption', 'Died', 'Euthanasia', 'Return_to_owner', 'Transfer')

# attach ID column
xgb_preds['ID'] <- Xs_test['ID']

write.csv(xgb_preds, './predictions/XGB.csv', row.names=FALSE)


#visualize feature importance
importance_matrix <- xgb.importance(factorVars, model = xgb_model)
xgb.plot.importance(importance_matrix)

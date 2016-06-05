numProcs = 4
addprocs(numProcs)

using DataFrames
using DecisionTree

#Load the data
Xs_train = convert(Array, readtable("./clean_data/Xs_train.csv"))
ys_train = convert(Array, readtable("./clean_data/ys_train.csv"))[:]
Xs_test = convert(Array, readtable("./clean_data/Xs_test.csv"))


#And start the actual learning
#acdc
model = RandomForestClassifier(nsubfeatures=0, ntrees=300, partialsampling=0.7)
DecisionTree.fit!(model, Xs_train, ys_train)

#Predict everything
probas = DecisionTree.predict_proba(model, Xs_test)


#And format to submit
submission = DataFrame(ID = 1:11456, Adoption = probas[:,1],Died = probas[:,2],Euthanasia = probas[:,3],Return_to_owner = probas[:,4],Transfer = probas[:,5])
writetable("./to_submit.csv", submission)

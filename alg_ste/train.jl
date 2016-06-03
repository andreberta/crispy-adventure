numProcs = 4
addProcs(numProcs)

using DataFrames
using DecisionTree

convert(Array)
#Load the data
Xs_train = convert(Array, readtable("./clean_data/Xs_train.csv"))
ys_train = convert(Array, readtable("./clean_data/ys_train.csv"))[:]
Xs_test = convert(Array, readtable("./clean_data/Xs_test.csv"))


#And start the actual learning
#acdc
model = build_forest(ys_train, Xs_train, 10, 100)
model_2 = nfoldCV_forest(ys_train, Xs_train, 2, 10, 5) #OOB come cazzo si fa?

#Predict everything
probas = apply_forest_proba(model, Xs_test, ["Adoption", "Died", "Euthanasia", "Return_to_owner", "Transfer"])


#And format to submit
submission = DataFrame(ID = 1:11456, Adoption = probas[:,1],Died = probas[:,2],Euthanasia = probas[:,3],Return_to_owner = probas[:,4],Transfer = probas[:,5])
writetable("./to_submit.csv", submission)

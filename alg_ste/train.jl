numProcs = 4
addProcs(numProcs)

using JLD
using DecisionTree

#Load the data
filename = "./clean_data.jld"
Xs_train = load(filename, "Xs_train")
ys_train = load(filename, "ys_train")
xs_train = load(filename, "xs_test")

#And start the actual learning
#acdc
model = build_forest(ys_train, Xs_train, 10, 100)


#Predict everything
probas = apply_forest_proba(model, Xs_test, ["Adoption", "Died", "Euthanasia", "Return_to_owner", "Transfer"])


#And format to submit
submission = DataFrame(ID = 1:11456, Adoption = probas[:,1],Died = probas[:,2],Euthanasia = probas[:,3],Return_to_owner = probas[:,4],Transfer = probas[:,5])
writetable("./to_submit.csv", submission)

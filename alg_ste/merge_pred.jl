using DataFrames

preds = readdir("./predictions/")

result = DataFrame()

for p in preds
    result = vcat(result, readtable(string("./predictions/",p)))
end

#just to reorder
result = result[[:ID, :Adoption, :Died, :Euthanasia, :Return_to_owner, :Transfer]]

#aggreagate various predictions
result = aggregate(result, :ID, mean)

#and rename
names!(result, [:ID, :Adoption, :Died, :Euthanasia, :Return_to_owner, :Transfer])

#save
writetable("./predictions_merged.csv", result)

using DataFrames
using DecisionTree

#Read the data
train = readtable("../data/train.csv")
#AnimalID, Name, DateTime, OutcomeType, OutcomeSubtype, AnimalType, SexuponOutcome, AgeuponOutcome, Breed, Color
test = readtable("../data/test.csv")
#ID, Name, DateTime, AnimalType, SexuponOutcome, AgeuponOutcome, Breed, Color


#put them together to simplify the code
full = vcat(train, test)

#Now we convert the AgeuponOutcome to days
function date_str_to_int(str::UTF8String) #start with "1 week"
    value = parse(Int, split(str, ' ')[1]) #this is 1
    unit = split(str, ' ')[2] #this is "week"
    unit = replace(unit, 's', "") #if it is "weeks" we drop the 's'

    if unit == "day"
        multiplier = 1
    elseif unit == "week"
        multiplier = 7
    elseif unit == "month"
        multiplier = 30
    elseif unit == "year"
        multiplier = 365
    else
        multiplier = NA
    end

    return value * multiplier
end

function date_str_to_int(na::NAtype)
    return NA
end

full[:AgeinDays] = map(date_str_to_int, full[:AgeuponOutcome])


# Replace blank names with "Nameless"
full[isna(full[:Name]), :Name] = "Nameless"


# Make a name v. no name variable
full[:HasName] = 1
full[full[:Name] .== "Nameless", :HasName] = 0


# Replace blank sex with "Unknown" (there is just one case in train, the original post replaces with the most common, both since we already haev some "Unknown" we better use them)
full[isna(full[:SexuponOutcome]), :SexuponOutcome] = "Unknown"


# Extract time variables from date
full[:DateTime] = DateTime(full[:DateTime], "yyyy-mm-dd HH:MM:SS")

full[:Hour] = Dates.hour(full[:DateTime])
full[:Weekday] = Dates.dayofweek(full[:DateTime])
full[:Month] = Dates.month(full[:DateTime])
full[:Year] = Dates.year(full[:DateTime])


# Time of day may also be useful
function time_of_day(hour::Int64)
    if hour > 5 && hour < 11 return "morning"
    elseif hour > 10 && hour < 16 return "midday"
    elseif hour > 15 && hour < 20 return "lateday"
    end
    return "night"
end

full[:TimeofDay] = map(time_of_day, full[:Hour])


# Now we take care of the breeds
# Find mixes
full[:IsMix] = 0
full[Bool[ismatch(r"Mix", x) for x in full[:Breed]], :IsMix] = 1

#Remove the word mix and split on / keeping only the first one (why?)
full[:SimpleBreed] = map(b -> replace(b, " Mix", ""), full[:Breed])
full[:SimpleBreed] = map(b -> split(b, "/")[1], full[:SimpleBreed])


# Now simplify the colors
full[:SimpleColor] = map(c -> split(c, "/")[1], full[:Color])
full[:SimpleColor] = map(c -> split(c, " ")[1], full[:SimpleColor])


# In the sex we have both sex and intactness, which are two different things
full[:Intact] = 0
full[Bool[ismatch(r"Intact", x) for x in full[:SexuponOutcome]], :Intact] = 1
full[Bool[ismatch(r"Unknown", x) for x in full[:SexuponOutcome]], :Intact] = 3 #on the forum they use "Unknown"

full[:Sex] = "Male"
full[Bool[ismatch(r"Female", x) for x in full[:SexuponOutcome]], :Sex] = "Female"
full[Bool[ismatch(r"Unknown", x) for x in full[:SexuponOutcome]], :Sex] = "Unknown"


# Now in AgeinDays we have some missing values, they impute them by using a decision tree, we just use the average
mean_age = mean(full[!isna(full[:AgeinDays]),:AgeinDays])
full[isna(full[:AgeinDays]),:AgeinDays] = mean_age


# Separate baby animals from grown up
full[:LifeStage] = "adult"
full[full[:AgeinDays] .<= 365, :LifeStage] = "baby"


#OK, now save train and test
training = [:Name, :AnimalType, :AgeinDays, :HasName, :Hour, :Weekday, :Month, :Year, :TimeofDay, :IsMix, :SimpleBreed, :SimpleColor, :Intact, :Sex, :LifeStage]
Xs_train = convert(Array, full[1:26729, training])
ys_train = convert(Array, full[1:26729, :OutcomeType])
Xs_test = convert(Array, full[26730:end, training])


#And start the actual learning
#acdc
model = build_forest(ys_train, Xs_train, 2, 10, 0.5)


#Predict everything
probas = apply_forest_proba(model, Xs_test, ["Adoption", "Died", "Euthanasia", "Return_to_owner", "Transfer"])


#And format to submit
submission = DataFrame(ID = 1:11456, Adoption = probas[:,1],Died = probas[:,2],Euthanasia = probas[:,3],Return_to_owner = probas[:,4],Transfer = probas[:,5])
writetable("./to_submit.csv", submission)

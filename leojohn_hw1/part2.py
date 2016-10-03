from decision_tree import DecisionTree
import pandas as pd
datasets ={"heart" :["heart_train.arff", "heart_test.arff"],
        "diabetes" :["diabetes_train.arff", "diabetes_test.arff"],   
       }
percentages =[5,10,20,50,100]
results =[]
m=4
data =[]
for k,v in datasets.iteritems(): 
    for percent in percentages:
        for iteration in xrange(1,11):
            decision_tree = DecisionTree(m,v[0],v[1],percent)
            decision_tree.fit()
            decision_tree.predict()
            data.append({"Iteration":iteration,"Dataset":k,"percentage":percent,"accuracy":decision_tree.accuracy}) 
dataframe = pd.DataFrame(data)
dataframe.to_csv("part2.csv",index=False)

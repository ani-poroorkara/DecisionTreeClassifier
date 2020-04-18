# Load the data as dataframe
import pandas as pd
d = pd.read_csv('dataset/graduate_admissions.csv')
len(d)


#Convert the Chance of Admit col into binary with 1 - admit and 0 - Reject
d['admit'] = d.apply(lambda row: 1 if (row['Chance of Admit ']) >= 0.75 else 0, axis=1)

#Drop Unwanted columns
d = d.drop(['Serial No.'], axis=1)

#Drop Chance of Admit as we have already benefited from it 
d = d.drop(['Chance of Admit '], axis=1)


#Randomly shuffle all the rows for random sampling.
#Shuffling data before learning has many benefits though it is not necessary
d = d.sample(frac = 1)

#Split into training and testing data 
# 80/20 : train/test
d_train = d[:400]
d_test = d[400:]

#remove the admit col and save separately for cross validation from all of the datasets
d_train_att = d_train.drop(['admit'], axis=1)
d_train_admit = d_train['admit']

d_test_att = d_test.drop(['admit'], axis=1)
d_test_admit = d_test['admit']

d_att = d.drop(['admit'], axis=1)
d_admit = d['admit']


#Perform an analysis of the dataset to find if its balanced or not
import numpy as np
adm = np.sum(d_admit)
tot = len(d_admit)
perofadm = str((adm/tot)*100)
print("Admitted Students:" + perofadm + "%")


#Creating the decision tree
from sklearn.model_selection import cross_val_score
from sklearn import tree

#Find the best depth of the classifier by creating trees and immediately cross validating 
for maxdepth in range (1,20):    
    t = tree.DecisionTreeClassifier (criterion ="entropy", max_depth = maxdepth)
    scores = cross_val_score(t, d_att, d_admit, cv = 5)
    print("Max depth: %d, Accuracy: %0.2f (+/- %0.2f)" % (maxdepth, scores.mean(), scores.std() * 2))

#We get depth = 2 with highest accuracy of 88% 
#Do note that due to the random sampling of our dataset, there is a slight probability that the depth may vary by 1 
t = tree.DecisionTreeClassifier (criterion ="entropy", max_depth = 2)
t = t.fit(d_train_att, d_train_admit)

#Create a visual of the final graph to visualize it
import graphviz
dot_data = tree.export_graphviz(t, out_file=None, label="all", impurity=False, proportion=True,
                                feature_names=list(d_train_att), class_names=["fail", "pass"], 
                                filled=True, rounded=True)
graph = graphviz.Source(dot_data)
graph

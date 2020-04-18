# DecisionTreeClassifier
Create any kind of Decision Tree Classifier with any dataset. 
This example uses [Graduate Admission 2](https://www.kaggle.com/mohansacharya/graduate-admissions) available on Kaggle

## Working of a Decision Tree
Place the best attribute of the dataset at the root of the tree.
Split the training set into subsets. Subsets should be made in such a way that each subset contains data with the same value for an attribute.
Repeat step 1 and step 2 on each subset until you find leaf nodes in all the branches of the tree.

## Dependencies
```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
```

## Run the classifier
1. Clone the repository 
2. Install dependencies
3. Run the rfc.py 

### License
[Apache License 2.0](https://github.com/ani-poroorkara/RandomForestClassifier/blob/master/LICENSE)

##### I recommend using Google Colab or Jupyter notebooks to run the file cell by cell
##### Connect with me on [LinkedIn](https://www.linkedin.com/in/anirudh-poroorkara-34900017b/)

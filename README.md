# Decision Tree Classifier
Create any kind of Decision Tree Classifier with any dataset.
This example uses [Graduate Admission 2](https://www.kaggle.com/mohansacharya/graduate-admissions) available on Kaggle

## Working of a Decision Tree
Decision tree is a type of supervised learning algorithm that is mostly used in classification problems.
It follows the steps:
1. Finding the best attributes using gini or information gain. (We use information gain in this example)
2. Place the best attribute of the dataset at the root of the tree.
3. Split the training set into subsets. Subsets should be made in such a way that each subset contains data with the same value for an attribute.
4. Repeat step 2 and step 3 on each subset until you find leaf nodes in all the branches of the tree.

## Dependencies
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn import tree
```

## Run the classifier
1. Clone the repository 
2. Install dependencies
3. Run the dtc.py 

### License
[Apache License 2.0](https://github.com/ani-poroorkara/DecisionTreeClassifier/blob/master/LICENSE)

##### I recommend using Google Colab or Jupyter notebooks to run the file cell by cell
##### Connect with me on [LinkedIn](https://www.linkedin.com/in/anirudh-poroorkara-34900017b/)

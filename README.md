### Spotting Most Important Features

**What we'll be doing:**
* loading libraries and data,
* training a model,
* knowing how a tree is represented,
* plotting feature importance

#### Load libraries
```python
%matplotlib inline
import xgboost as xgb
import seaborn as sns
import pandas as pd
sns.set(font_scale = 1.5)
```
#### Load data
```python
dtrain = xgb.DMatrix('../data/agaricus.txt.train')
dtest = xgb.DMatrix('../data/agaricus.txt.test')
```
#### Train the model
```python
# specify training parameters
params = {
    'objective':'binary:logistic',
    'max_depth':1,
    'silent':1,
    'eta':0.5
}

num_rounds = 5
```
Using  5 stump decision trees with average learning rate.

Train the model. 
```python
# see how does it perform
watchlist  = [(dtest,'test'), (dtrain,'train')] # native interface only
```
In the same time specify watchlist to observe it's performance on the test set.
```python
bst = xgb.train(params, dtrain, num_rounds, watchlist)
```
#### Representation of a tree
> While building a tree is divided recursively several times (in this example only once) - this 
> operation is called **split**. To perform a split the algorithm must figure out which is the best (one) 
> feature to use.
> After that, at the bottom of the we get groups of observations packed in the leaves.

> In the final model, these leafs are supposed to be as pure as possible for each tree, meaning in our 
> case that each leaf should be made of one label class.

> Not all splits are equally important. Basically the first split of a tree will have more impact on 
> the purity that, for instance, the deepest split. Intuitively, we understand that the first split 
> makes most of the work, and the following splits focus on smaller parts of the dataset which have 
> been missclassified by the first tree.

> In the same way, in Boosting we try to optimize the missclassification at each round (it is called 
> the loss). So the first tree will do the big work and the following trees will focus on the remaining, 
> on the parts not correctly learned by the previous trees.

> The improvement brought by each split can be measured, it is the gain.

> ~ Quoted from the Kaggle Tianqi Chen's Kaggle [notebook](https://www.kaggle.com/tqchen/understanding-xgboost-model-on-otto-data).

Let's investigate how trees look like on our case:
```python
trees_dumptrees_d  = bst.get_dump(fmap='../data/featmap.txt', with_stats=True)

for tree in trees_dump:
    print(tree)
```
For each split we are getting the following details:

* which feature was used to make split,
* possible choices to make (branches)
* **gain** which is the actual improvement in accuracy brough by that feature. The idea is that before 
adding a new split on a feature X to the branch there was some wrongly classified elements, after adding 
the split on this feature, there are two new branches, and each of these branch is more accurate (one 
branch saying if your observation is on this branch then it should be classified as 1, and the other 
branch saying the exact opposite),
* **cover** measuring the relative quantity of observations concerned by that feature

#### Plotting
Use built-in function *plot_importance* to create a plot presenting most important features due to some criterias. 
```python
xgb.plot_importance(bst, importance_type='gain', xlabel='Gain')
```
We can simplify it a little bit by introducing a *F-score* metric.

> **F-score** - sums up how many times a split was performed on each feature.
```python
xgbxgb..plot_importanceplot_im (bst)
```
Created model is convinient to access F-score by *get_fscore()*:
```python
importancesimporta  = bst.get_fscore()
importances
```
Manipulate data in your own way
```python
# create df
importance_df = pd.DataFrame({
        'Splits': list(importances.values()),
        'Feature': list(importances.keys())
    })
importance_df.sort_values(by='Splits', inplace=True)
importance_df.plot(kind='barh', x='Feature', figsize=(8,6), color='orange')
```








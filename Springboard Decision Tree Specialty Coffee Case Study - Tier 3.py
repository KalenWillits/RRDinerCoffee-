# %% markdown
# # **Springboard Decision Tree Specialty Coffee Case Study - Tier 3**
#
#
#
# %% markdown
# # The Scenario
#
# Imagine you've just finished the Springboard Data Science Career Track course, and have been hired by a rising popular specialty coffee company - RR Diner Coffee - as a data scientist. Congratulations!
#
# RR Diner Coffee sells two types of item:
# - specialty coffee beans, in bulk (by the kilogram only)
# - coffee equipment and merchandise (grinders, brewing equipment, mugs, books, t-shirts)
#
# RR Diner Coffee has three stores, two in Europe and one in the United States. The flagshap store is in the US, and everything is quality assessed there, before being shipped out. Customers further away from the US flagship store have higher shipping charges.
#
# You've been taken on at RR Diner Coffee because the company is turning towards using data science and machine learning to systematically make decisions about which coffee farmers they should strike deals with.
#
# RR Diner Coffee typically buys coffee from farmers, processes it on site, brings it back to the US, roasts it, packages it, markets it, and ships it (only in bulk, and after quality assurance) to customers internationally. These customers all own coffee shops in major cities like New York, Paris, London, Hong Kong, Tokyo, and Berlin.
#
# Now, RR Diner Coffee has a decision about whether to strike a deal with a legendary coffee farm (known as the **Hidden Farm**) in rural China: there are rumors their coffee tastes of lychee and dark chocolate, while also being as sweet as apple juice.
#
# It's a risky decision, as the deal will be expensive, and the coffee might not be bought by customers. The stakes are high: times are tough, stocks are low, farmers are reverting to old deals with the larger enterprises and the publicity of selling *Hidden Farm* coffee could save the RR Diner Coffee business.
#
# Your first job, then, is ***to build a decision tree to predict how many units of the Hidden Farm Chinese coffee will be purchased by RR Diner Coffee's most loyal customers.***
#
# To this end, you and your team have conducted a survey of 710 of the most loyal RR Diner Coffee customers, collecting data on the customers':
# - age
# - gender
# - salary
# - whether they have bought at least one RR Diner Coffee product online
# - their distance from the flagship store in the US (standardized to a number between 0 and 11)
# - how much they spent on RR Diner Coffee products on the week of the survey
# - how much they spent on RR Diner Coffee products in the month preceding the survey
# - the number of RR Diner coffee bean shipments each customer has ordered over the preceding year.
#
# You also asked each customer participating in the survey whether they would buy the Hidden Farm coffee, and some (but not all) of the customers gave responses to that question.
#
# You sit back and think: if more than 70% of the interviewed customers are likely to buy the Hidden Farm coffee, you will strike the deal with the local Hidden Farm farmers and sell the coffee. Otherwise, you won't strike the deal and the Hidden Farm coffee will remain in legends only. There's some doubt in your mind about whether 70% is a reasonable threshold, but it'll do for the moment.
#
# To solve the problem, then, you will build a decision tree to implement a classification solution.
#
#
# -------------------------------
# As with other case studies in this course, this notebook is **tiered**, meaning you can elect the tier that is right for your confidence and skill level. There are 3 tiers, with tier 1 being the easiest and tier 3 being the hardest. This is ***tier 3***.
#
# **1. Sourcing and loading**
# - Import packages
# - Load data
# - Explore the data
#
#
# **2. Cleaning, transforming, and visualizing**
# - Cleaning the data
# - Train/test split
#
#
# **3. Modeling**
# - Model 1: Entropy model - no max_depth
# - Model 2: Gini impurity model - no max_depth
# - Model 3: Entropy model - max depth 3
# - Model 4: Gini impurity model - max depth 3
#
#
# **4. Evaluating and concluding**
# - How many customers will buy Hidden Farm coffee?
# - Decision
#
# **5. Random Forest**
# - Import necessary modules
# - Model
# - Revise conclusion
#
# %% markdown
# # 0. Overview
#
# This notebook uses decision trees to determine whether the factors listed above (salary, gender, age, how much money the customer spent last week and during the preceding month on RR Diner Coffee products, how many kilogram coffee bags the customer bought over the last year, whether they have bought at least one RR Diner Coffee product online, and their distance from the flagship store in the USA), could predict whether customers would purchase the Hidden Farm coffee if a deal with its farmers were struck.
# %% markdown
# # 1. Sourcing and loading
# ## 1a. Import Packages
# %% codecell
import pandas as pd
import numpy as np
from sklearn import tree, metrics
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.externals.six import StringIO
from IPython.display import Image
import pydotplus
# %% markdown
# ## 1b. Load data
# %% codecell
# Read in the data to a variable called coffeeData
cd_data = 'data/'
data = 'RRDinerCoffeeData.csv'
coffeeData = pd.read_csv(cd_data + data)

# %% markdown
# ## 1c. Explore the data
# %% markdown
# As we've seen, exploration entails doing things like checking out the **initial appearance** of the data with head(), the **dimensions** of our data with .shape, the **data types** of the variables with .info(), the **number of non-null values**, how much **memory** is being used to store the data, and finally the major summary statistcs capturing **central tendancy, dispersion and the null-excluding shape of the dataset's distribution**.
#
# How much of this can you do yourself by this point in the course? Have a real go.
# %% codecell
# Call head() on your data
coffeeData.head()
# %% codecell
# Call .shape on your data
coffeeData.shape
# %% codecell
# Call info() on your data
coffeeData.info()
# %% codecell
# Call describe() on your data to get the relevant summary statistics for your data
coffeeData.describe()
# %% markdown
# # 2. Cleaning, transforming, and visualizing
# ## 2a. Cleaning the data
# %% markdown
# Some datasets don't require any cleaning, but almost all do. This one does. We need to replace '1.0' and '0.0' in the 'Decision' column by 'YES' and 'NO' respectively, clean up the values of the 'gender' column, and change the column names to words which maximize meaning and clarity.
# %% markdown
# First, let's change the name of `spent_week`, `spent_month`, and `SlrAY` to `spent_last_week` and `spent_last_month` and `salary` respectively.
# %% codecell
# Check out the names of our data's columns
coffeeData.columns

# %% codecell
# Make the relevant name changes to spent_week and spent_per_week.
rename_map = {
    'spent_week':'spent_last_week',
    'spent_month': 'spent_last_month',
    'SlrAY': 'salary'
}
coffeeData = coffeeData.rename(rename_map, axis=1)
# %% codecell
# Check out the column names
coffeeData.columns
# %% codecell
# Let's have a closer look at the gender column. Its values need cleaning.
coffeeData['Gender'].isna().sum()
# %% codecell
# See the gender column's unique values
coffeeData['Gender'].unique()

alt_female = ['female', 'F', 'f ', 'FEMALE']
alt_male =  ['MALE', 'male', 'M']
# %% markdown
# We can see a bunch of inconsistency here.
#
# Use replace() to make the values of the `gender` column just `Female` and `Male`.
# %% codecell
# Replace all alternate values for the Female entry with 'Female'
coffeeData['Gender'].replace(to_replace=alt_female, value='Female', inplace=True)
# %% codecell
# Check out the unique values for the 'gender' column
coffeeData['Gender'].unique()

# %% codecell
coffeeData['Gender'].replace(to_replace=alt_male, value='Male', inplace=True)

# %% codecell
# Let's check the unique values of the column "gender"
coffeeData['Gender'].unique()

# %% codecell
# Check out the unique values of the column 'Decision'
coffeeData['Decision'].unique()
# %% markdown
# We now want to replace `1.0` and `0.0` in the `Decision` column by `YES` and `NO` respectively.
# %% codecell
# Replace 'Yes' and 'No' by 1 and 0
coffeeData['Decision'].replace(to_replace=1.0, value='YES', inplace=True)
coffeeData['Decision'].replace(to_replace=0.0, value='NO', inplace=True)

# %% codecell
# Check that our replacing those values with 'YES' and 'NO' worked, with unique()
coffeeData['Decision'].unique()
# %% markdown
# ## 2b. Train/test split
# To execute the train/test split properly, we need to do five things:
# 1. Drop all rows with a null value in the `Decision` column, and save the result as NOPrediction: a dataset that will contain all known values for the decision
# 2. Visualize the data using scatter and boxplots of several variables in the y-axis and the decision on the x-axis
# 3. Get the subset of coffeeData with null values in the `Decision` column, and save that subset as Prediction
# 4. Divide the NOPrediction subset into X and y, and then further divide those subsets into train and test subsets for X and y respectively
# 5. Create dummy variables to deal with categorical inputs
# %% markdown
# ### 1. Drop all null values within the `Decision` column, and save the result as NoPrediction
# %% codecell
# NoPrediction will contain all known values for the decision
# Call dropna() on coffeeData, and store the result in a variable NOPrediction
# Call describe() on the Decision column of NoPrediction after calling dropna() on coffeeData
NoPrediction = coffeeData.dropna()
NoPrediction['Decision'].describe()
# %% markdown
# ### 2. Visualize the data using scatter and boxplots of several variables in the y-axis and the decision on the x-axis
# %% codecell
# Exploring our new NOPrediction dataset
# Make a boxplot on NOPrediction where the x axis is Decision, and the y axis is spent_last_week
sns.boxplot(x=NoPrediction['Decision'], y=NoPrediction['spent_last_week'])
plt.title('Boxplot_DecisionVSspent_last_week')
plt.savefig('figures/Boxplot_DecisionVSspent_last_week.png')
plt.show()
# %% markdown
# Can you admissibly conclude anything from this boxplot? Write your answer here:
# The variance of what was spent is higher in the customers that answered no.
# Also, the customers that answered yes tend to make more expensive purchases.
# %% codecell
# Make a scatterplot on NOPrediction, where x is distance, y is spent_last_month and hue is Decision
sns.scatterplot(x=NoPrediction['Distance'], y=NoPrediction['spent_last_month'], hue=NoPrediction['Decision'])
plt.title('scatterplot_DistanceVSspent_last_month')
plt.savefig('figures/scatterplot_DistanceVSspent_last_month.png')
plt.show()
# %% markdown
# Can you admissibly conclude anything from this scatterplot? Remember: we are trying to build a tree to classify unseen examples. Write your answer here:
# There is a clean desiscion boundry where the customers who are further away and spent less answered yes.
# This feature seems to scale almost linearly.
# %% markdown
# ### 3. Get the subset of coffeeData with null values in the Decision column, and save that subset as Prediction
# %% codecell
# Get just those rows whose value for the Decision column is null
Prediction = coffeeData[pd.isnull(coffeeData["Decision"])]

#coffeeData['Decision'].replace(to_replace=np.NaN, value=0, inplace=True)
#Prediction = coffeeData[coffeeData['Decision'] == 0]['Decision']
Prediction.head()
# %% codecell
# Call describe() on Prediction
Prediction.describe()

# %% markdown
# ### 4. Divide the NOPrediction subset into X and y
# %% codecell
# Check the names of the columns of NOPrediction
NoPrediction.columns
# %% codecell
# Let's do our feature selection.
# Make a variable called 'features', and a list containing the strings of every column except "Decision"

features = ['Age', 'Gender', 'num_coffeeBags_per_year', 'spent_last_week',
       'spent_last_month', 'salary', 'Distance', 'Online']

# Make an explanatory variable called X, and assign it: NoPrediction[features]
X = NoPrediction[features]

# Make a dependent variable called y, and assign it: NoPrediction.Decision
y = NoPrediction['Decision']
# %% markdown
# ### 4. Further divide those subsets into train and test subsets for X and y respectively: X_train, X_test, y_train, y_test
# %% codecell
# Call train_test_split on X, y. Make the test_size = 0.25, and random_state = 246
X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.25, random_state=246)
# %% markdown
# ### 5. Create dummy variables to deal with categorical inputs
# One-hot encoding replaces each unique value of a given column with a new column, and puts a 1 in the new column for a given row just if its initial value for the original column matches the new column. Check out [this resource](https://hackernoon.com/what-is-one-hot-encoding-why-and-when-do-you-have-to-use-it-e3c6186d008f) if you haven't seen one-hot-encoding before.
# %% codecell
# One-hot encode all features in training set.
X_train = pd.get_dummies(X_train, dtype=float)
# y_train = pd.get_dummies(y_train)
# Do the same, but for X_test
X_test = pd.get_dummies(X_test)
# %% markdown
# # 3. Modeling
# It's useful to look at the scikit-learn documentation on decision trees https://scikit-learn.org/stable/modules/tree.html before launching into applying them. If you haven't seen them before, take a look at that link, in particular the section `1.10.5.`
# %% markdown
# ## Model 1: Entropy model - no max_depth
#
# We'll give you a little more guidance here, as the Python is hard to deduce, and scikitlearn takes some getting used to.
#
# Theoretically, let's remind ourselves of what's going on with a decision tree implementing an entropy model.
#
# Ross Quinlan's **ID3 Algorithm** was one of the first, and one of the most basic, to use entropy as a metric.
#
# **Entropy** is a measure of how uncertain we are about which category the data-points fall into at a given point in the tree. The **Information gain** of a specific feature with a threshold (such as 'spent_last_month <= 138.0') is the difference in entropy that exists before and after splitting on that feature; i.e., the information we gain about the categories of the data-points by splitting on that feature and that threshold.
#
# Naturally, we want to minimize entropy and maximize information gain. Quinlan's ID3 algorithm is designed to output a tree such that the features at each node, starting from the root, and going all the way down to the leaves, have maximial information gain. We want a tree whose leaves have elements that are *homogeneous*, that is, all of the same category.
#
# The first model will be the hardest. Persevere and you'll reap the rewards: you can use almost exactly the same code for the other models.
# %% codecell
# Declare a variable called entr_model and use tree.DecisionTreeClassifier.
y_train.head()
entr_model = tree.DecisionTreeClassifier(criterion="entropy", random_state = 1234)
# Call fit() on entr_model
y_train = y_train.astype('float64')
entr_model.fit(X_train, y_train)
X_train.head()
# Call predict() on entr_model with X_test passed to it, and assign the result to a variable y_pred
_ _ _

# Call Series on our y_pred variable with the following: pd.Series(y_pred)
_ _ _

# Check out entr_model
entr_model
# %% codecell
# Now we want to visualize the tree
_ _ _

# We can do so with export_graphviz
_ _ _

# Alternatively for class_names use entr_model.classes_
_ _ _
# %% markdown
# ## Model 1: Entropy model - no max_depth: Interpretation and evaluation
# %% codecell
# Run this block for model evaluation metrics
print("Model Entropy - no max depth")
print("Accuracy:", metrics.accuracy_score(y_test,y_pred))
print("Balanced accuracy:", metrics.balanced_accuracy_score(y_test,y_pred))
print('Precision score for "Yes"' , metrics.precision_score(y_test,y_pred, pos_label = "YES"))
print('Precision score for "No"' , metrics.precision_score(y_test,y_pred, pos_label = "NO"))
print('Recall score for "Yes"' , metrics.recall_score(y_test,y_pred, pos_label = "YES"))
print('Recall score for "No"' , metrics.recall_score(y_test,y_pred, pos_label = "NO"))
# %% markdown
# What can you infer from these results? Write your conclusions here:
# %% markdown
# ## Model 2: Gini impurity model - no max_depth
#
# Gini impurity, like entropy, is a measure of how well a given feature (and threshold) splits the data into categories.
#
# Their equations are similar, but Gini impurity doesn't require logarithmic functions, which can be computationally expensive.
# %% codecell
# Make a variable called gini_model, and assign it exactly what you assigned entr_model with above, but with the
# criterion changed to 'gini'
_ _ _

# Call fit() on the gini_model as you did with the entr_model
_ _ _

# Call predict() on the gini_model as you did with the entr_model
_ _ _

# Turn y_pred into a series, as before
_ _ _

# Check out gini_model
_ _ _
# %% codecell
# As before, but make the model name gini_model
_ _ _
_ _ _

# Alternatively for class_names use gini_model.classes_
_ _ _
# %% codecell
# Run this block for model evaluation
print("Model Gini impurity model")
print("Accuracy:", metrics.accuracy_score(y_test,y_pred))
print("Balanced accuracy:", metrics.balanced_accuracy_score(y_test,y_pred))
print('Precision score' , metrics.precision_score(y_test,y_pred, pos_label = "YES"))
print('Recall score' , metrics.recall_score(y_test,y_pred, pos_label = "NO"))
# %% markdown
# How do the results here compare to the previous model? Write your judgments here:
# %% markdown
# ## Model 3: Entropy model - max depth 3
# We're going to try to limit the depth of our decision tree, using entropy first.
#
# As you know, we need to strike a balance with tree depth.
#
# Insufficiently deep, and we're not giving the tree the opportunity to spot the right patterns in the training data.
#
# Excessively deep, and we're probably going to make a tree that overfits to the training data, at the cost of very high error on the (hitherto unseen) test data.
#
# Sophisticated data scientists use methods like random search with cross-validation to systematically find a good depth for their tree. We'll start with picking 3, and see how that goes.
# %% codecell
# Made a model as before, but call it entr_model2, and make the max_depth parameter equal to 3.
# Execute the fitting, predicting, and Series operations as before
_ _ _
# %% codecell
# As before, we need to visualize the tree to grasp its nature
_ _ _

# Alternatively for class_names use entr_model2.classes_
_ _ _
# %% codecell
# Run this block for model evaluation
print("Model Entropy model max depth 3")
print("Accuracy:", metrics.accuracy_score(y_test,y_pred))
print("Balanced accuracy:", metrics.balanced_accuracy_score(y_test,y_pred))
print('Precision score for "Yes"' , metrics.precision_score(y_test,y_pred, pos_label = "YES"))
print('Recall score for "No"' , metrics.recall_score(y_test,y_pred, pos_label = "NO"))
# %% markdown
# So our accuracy decreased, but is this certainly an inferior tree to the max depth original tree we did with Model 1? Write your conclusions here:
# %% markdown
# ## Model 4: Gini impurity  model - max depth 3
# We're now going to try the same with the Gini impurity model.
# %% codecell
# As before, make a variable, but call it gini_model2, and ensure the max_depth parameter is set to 3
_ _ _ = _ _ _._ _ _(_ _ _ ='_ _ _ ', _ _ _ = 1234, _ _ _  = _ _ _ )

# Do the fit, predict, and series transformations as before.
_ _ _
# %% codecell
dot_data = StringIO()
_ _ _


# Alternatively for class_names use gini_model2.classes_
_ _ _
# %% codecell
print("Gini impurity  model - max depth 3")
print("Accuracy:", metrics.accuracy_score(y_test,y_pred))
print("Balanced accuracy:", metrics.balanced_accuracy_score(y_test,y_pred))
print('Precision score' , metrics.precision_score(y_test,y_pred, pos_label = "YES"))
print('Recall score' , metrics.recall_score(y_test,y_pred, pos_label = "NO"))
# %% markdown
# Now this is an elegant tree. Its accuracy might not be the highest, but it's still the best model we've produced so far. Why is that? Write your answer here:
# %% markdown
# # 4. Evaluating and concluding
# ## 4a. How many customers will buy Hidden Farm coffee?
# Let's first ascertain how many loyal customers claimed, in the survey, that they will purchase the Hidden Farm coffee.
# %% codecell
# Call value_counts() on the 'Decision' column of the original coffeeData
_ _ _
# %% markdown
# Let's now determine the number of people that, according to the model, will be willing to buy the Hidden Farm coffee.
# 1. First we subset the Prediction dataset into `new_X` considering all the variables except `Decision`
# 2. Use that dataset to predict a new variable called `potential_buyers`
# %% codecell
# Feature selection
# Make a variable called feature_cols, and assign it a list containing all the column names except 'Decision'
_ _ _

# Make a variable called new_X, and assign it the subset of Prediction, containing just the feature_cols
_ _ _
# %% codecell
# Call get_dummies() on the Pandas object pd, with new_X plugged in, to one-hot encode all features in the training set
_ _ _

# Make a variable called potential_buyers, and assign it the result of calling predict() on a model of your choice;
# don't forget to pass new_X to predict()
_ _ _
# %% codecell
# Let's get the numbers of YES's and NO's in the potential buyers
# Call unique() on np, and pass potential_buyers and return_counts=True
_ _ _
# %% markdown
# The total number of potential buyers is 303 + 183 = 486
# %% codecell
# Print the total number of surveyed people
_ _ _
# %% codecell
# Let's calculate the proportion of buyers
_ _ _
# %% codecell
# Print the percentage of people who want to buy the Hidden Farm coffee, by our model
_ _ _
# %% markdown
# ## 4b. Decision
# Remember how you thought at the start: if more than 70% of the interviewed customers are likely to buy the Hidden Farm coffee, you will strike the deal with the local Hidden Farm farmers and sell the coffee. Otherwise, you won't strike the deal and the Hidden Farm coffee will remain in legends only. Well now's crunch time. Are you going to go ahead with that idea? If so, you won't be striking the deal with the Chinese farmers.
#
# They're called `decision trees`, aren't they? So where's the decision? What should you do? (Cue existential cat emoji).
#
# Ultimately, though, we can't write an algorithm to actually *make the business decision* for us. This is because such decisions depend on our values, what risks we are willing to take, the stakes of our decisions, and how important it us for us to *know* that we will succeed. What are you going to do with the models you've made? Are you going to risk everything, strike the deal with the *Hidden Farm* farmers, and sell the coffee?
#
# The philosopher of language Jason Stanley once wrote that the number of doubts our evidence has to rule out in order for us to know a given proposition depends on our stakes: the higher our stakes, the more doubts our evidence has to rule out, and therefore the harder it is for us to know things. We can end up paralyzed in predicaments; sometimes, we can act to better our situation only if we already know certain things, which we can only if our stakes were lower and we'd *already* bettered our situation.
#
# Data science and machine learning can't solve such problems. But what it can do is help us make great use of our data to help *inform* our decisions.
# %% markdown
# ## 5. Random Forest
# You might have noticed an important fact about decision trees. Each time we run a given decision tree algorithm to make a prediction (such as whether customers will buy the Hidden Farm coffee) we will actually get a slightly different result. This might seem weird, but it has a simple explanation: machine learning algorithms are by definition ***stochastic***, in that their output is at least partly determined by randomness.
#
# To account for this variability and ensure that we get the most accurate prediction, we might want to actually make lots of decision trees, and get a value that captures the center or average of the outputs of those trees. Luckily, there's a method for this, known as the ***Random Forest***.
#
# Essentially, Random Forest involves making lots of trees with similar properties, and then performing summary statistics on the outputs of those trees to reach that central value. Random forests are hugely powerful classifers, and they can improve predictive accuracy and control over-fitting.
#
# Why not try to inform your decision with random forest? You'll need to make use of the RandomForestClassifier function within the sklearn.ensemble module, found [here](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html).
# %% markdown
# ### 5a. Import necessary modules
# %% codecell
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
# %% markdown
# ### 5b. Model
# You'll use your X_train and y_train variables just as before.
#
# You'll then need to make a variable (call it firstRFModel) to store your new Random Forest model. You'll assign this variable the result of calling RandomForestClassifier().
#
# Then, just as before, you'll call fit() on that firstRFModel variable, and plug in X_train and y_train.
#
# Finally, you should make a variable called y_pred, and assign it the result of calling the predict() method on your new firstRFModel, with the X_test data passed to it.
# %% codecell
# Plug in appropriate max_depth and random_state parameters
_ _ _

# Model and fit
_ _ _



# %% markdown
# ### 5c. Revise conclusion
#
# Has your conclusion changed? Or is the result of executing random forest the same as your best model reached by a single decision tree?

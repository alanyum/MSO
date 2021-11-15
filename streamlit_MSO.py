# streamlit_MSO.
# Reference:
# "DS_SPACE.MSO_USAGE_WEEK_SUBDEPT.ipynb"
# "streamlit_demo_v1.py"_index
# Output current location of directory.
#import os
#os.getcwd()

# Change current directory location.
#os.chdir('C:\\Users\\Alan.Yum\\OneDrive - A.S. Watson Group\\Projects\\Macro Space Optimization\\Jupyter')

# Import required packages.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import visualisation packages.
# Reference:
# https://www.dataquest.io/blog/tutorial-time-series-analysis-with-pandas/
# %%capture
#! pip install seaborn
import seaborn as sns
sns.set(style = "ticks", color_codes = True, rc = {"figure.figsize":(11, 4)}) # Use seaborn style defaults and set the default figure size
# Reference for purpose of the following usages: 
#%matplotlib inline 
from scipy import stats

import streamlit as st

st.title("Macro Space Optimization")
'''
## DS_SPACE.MSO_USAGE_WEEK_SUBDEPT
'''

st.subheader("Collect Initial Data")
df_mso_usage_week_subdept_all = pd.read_csv("HK_mso_usage_week_subdept.csv", low_memory = False)
df_mso_usage_week_subdept_all_100 = pd.read_csv("HK_mso_usage_week_subdept.csv", low_memory = False, nrows = 100)
st.write("Data size (before cleaning):")
df_mso_usage_week_subdept_all.shape
st.write(df_mso_usage_week_subdept_all_100)

st.subheader("Clean Data")
# Drop row with NA in column values.
df_mso_usage_week_subdept_all = df_mso_usage_week_subdept_all.dropna()
st.write("Data size (after cleaning):")
df_mso_usage_week_subdept_all.shape

st.subheader("Describe Data / Explore data")
# Highest "WEEKLY_SALES" by "LOC_KEY".
# Reference:
# https://www.kdnuggets.com/2021/04/e-commerce-data-analysis-sales-strategy-python.html
## Which store has the highest sales?
df_mso_usage_week_subdept_all_LOC_SALES = df_mso_usage_week_subdept_all.groupby(["LOC_KEY"])["WEEKLY_SALES"].sum().sort_values(ascending = False).to_frame()
st.write("Highest weekly sales by location:")
st.dataframe(df_mso_usage_week_subdept_all_LOC_SALES.style.highlight_max(axis = 0))

df_mso_usage_week_subdept_all_hol_ym_cpi = pd.read_csv("HK_mso_usage_week_subdept_hol_ym_cpi.csv")

# Highest "WEEKLY_SALES" by "LOC_KEY" and "SUBDEPT".
df_mso_usage_week_subdept_all_hol_ym_cpi_LOC_SUBDEPT_SALES = df_mso_usage_week_subdept_all_hol_ym_cpi.groupby(["LOC_KEY", "SUBDEPT"])["WEEKLY_SALES"].sum().sort_values(ascending = False).to_frame()
st.write("Highest weekly sales by location and subdepartment:")
st.dataframe(df_mso_usage_week_subdept_all_hol_ym_cpi_LOC_SUBDEPT_SALES.style.highlight_max(axis = 0))

# Read data: df_mso_usage_week_subdept_all_hol_ym_cpi_L1333 and df_mso_usage_week_subdept_all_hol_ym_cpi_L1333_sales_margins_gb_week.
# Reference:
# https://stackoverflow.com/questions/55240330/how-to-read-csv-file-from-github-using-pandas
url = "https://github.com/alanyum/MSO/blob/master/df_mso_usage_week_subdept_all_hol_ym_cpi_L1333.csv?raw=true"
df_mso_usage_week_subdept_all_hol_ym_cpi_L1333 = pd.read_csv(url, index_col = 0)
df_mso_usage_week_subdept_all_hol_ym_cpi_L1333_sales_margins_gb_week = pd.read_csv("df_mso_usage_week_subdept_all_hol_ym_cpi_L1333_sales_margins_gb_week.csv")

st.subheader("Store: 1333")
# Replicate the above time series plot of "WEEKLY_SALES", "Is_Holiday", "CPI" and "WEEKLY_SALES+7" by unqiue "WK_IDNT_new" of store's location: 1333
# with plotly.
# Reference:
# https://plotly.com/python/subplots/
# https://plotly.com/python/table-subplots/
# https://plotly.com/python-api-reference/generated/plotly.graph_objects.Scatter.html
from plotly.subplots import make_subplots
import plotly.graph_objects as go

fig = make_subplots(rows = 4, cols = 1)

fig.add_trace(
    go.Scatter(
        x = df_mso_usage_week_subdept_all_hol_ym_cpi_L1333_sales_margins_gb_week["WK_IDNT_new"],
        y = df_mso_usage_week_subdept_all_hol_ym_cpi_L1333_sales_margins_gb_week["WEEKLY_SALES"],
        mode = "markers",
        name = "Weekly Sales"
    ),
    row = 1, col = 1)

fig.add_trace(
    go.Scatter(
        x = df_mso_usage_week_subdept_all_hol_ym_cpi_L1333_sales_margins_gb_week["WK_IDNT_new"],
        y = df_mso_usage_week_subdept_all_hol_ym_cpi_L1333_sales_margins_gb_week["Is_Holiday"],
        mode = "markers",
        name = "Holiday"
    ),
    row = 2, col = 1)

fig.add_trace(
    go.Scatter(
        x = df_mso_usage_week_subdept_all_hol_ym_cpi_L1333_sales_margins_gb_week["WK_IDNT_new"],
        y = df_mso_usage_week_subdept_all_hol_ym_cpi_L1333_sales_margins_gb_week["CPI"],
        mode = "markers",
        name = "CPI"
    ),
    row = 3, col = 1)

## Aggregate "WEEKLY_SALES", "WEEKLY_GMARGIN" and "WEEKLY_TMARGIN" by "WK_IDNT_new" on data: df_mso_usage_week_subdept_all_hol_ym_cpi_L1333.
df_mso_usage_week_subdept_all_hol_ym_cpi_L1333_sales_7_gb_week = df_mso_usage_week_subdept_all_hol_ym_cpi_L1333.groupby(by = ["WK_IDNT_new"]).agg({"WEEKLY_SALES+7":"sum"})

## Reset index.
df_mso_usage_week_subdept_all_hol_ym_cpi_L1333_sales_7_gb_week.reset_index(inplace = True)

fig.add_trace(
    go.Scatter(
        x = df_mso_usage_week_subdept_all_hol_ym_cpi_L1333_sales_7_gb_week["WK_IDNT_new"],
        y = df_mso_usage_week_subdept_all_hol_ym_cpi_L1333_sales_7_gb_week["WEEKLY_SALES+7"],
        mode = "markers",
        name = "Weekly Sales of Previous 7-day"
    ),
    row = 4, col = 1)

fig.update_layout(height = 800, width = 800, title_text = "Time Series plots")
st.plotly_chart(fig, use_container_width = True)

# Perform ordinal encoder on data: df_mso_usage_week_subdept_all_hol_ym_cpi(1)
# Reference:
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder.html#sklearn.preprocessing.OrdinalEncoder
## Create copy of data: df_mso_usage_week_subdept_all_gb.
## Reference:
## https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.copy.html
df_mso_usage_week_subdept_all_hol_ym_cpi_ordenc = df_mso_usage_week_subdept_all_hol_ym_cpi.copy()

## Perform ordinal encoding on selected features: "POG_DIVISION_CODE", "POG_DEPT_CODE" and "SUBDEPT".
from sklearn import preprocessing
ordenc = preprocessing.OrdinalEncoder()
subX_ordenc = ordenc.fit_transform(df_mso_usage_week_subdept_all_hol_ym_cpi_ordenc[["POG_DIVISION_CODE", "POG_DEPT_CODE", "SUBDEPT", "Month", "Year-Month"]])

## Convert to data frame.
## Reference:
## https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html
df_mso_usage_week_subdept_all_hol_ym_cpi_ordenc[["POG_DIVISION_CODE", "POG_DEPT_CODE", "SUBDEPT", "Month", "Year-Month"]] = pd.DataFrame(data = subX_ordenc, columns = ["POG_DIVISION_CODE", "POG_DEPT_CODE", "SUBDEPT", "Month", "Year-Month"])

# Perform AdaBoost on data: df_mso_usage_week_subdept_all_hol_ym_cpi_ordenc of store: 1333(1):
# Target: "WEEKLY_SALES" by unique "WK_IDNT", "USAGE_TYPE", "POG_DIVISION_CODE", "POG_DEPT_CODE", "SUBDEPT", "ADJUSTED_SUBDEPT_METERAGE", "Year", "Month", "Is_Holiday" and "CPI"
# Reference:
# https://scikit-learn.org/stable/modules/ensemble.html#adaboost
from sklearn.datasets import make_classification
from sklearn.datasets import make_regression

## Split training and testing sets.
## Reference:
## https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split
## https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation
from sklearn.model_selection import train_test_split
#x, y = df_mso_usage_week_subdept_all_gb_ordenc[["USAGE_TYPE", "LOC_KEY", "POG_DIVISION_CODE", "POG_DEPT_CODE", "SUBDEPT",
#                                                "ADJUSTED_SUBDEPT_METERAGE", "Is_Holiday", "Year", "Month"]], df_mso_usage_week_subdept_all_gb_ordenc["WEEKLY_SALES"]
x, y = df_mso_usage_week_subdept_all_hol_ym_cpi_ordenc[df_mso_usage_week_subdept_all_hol_ym_cpi_ordenc["LOC_KEY"] == 1333][["WK_IDNT", "USAGE_TYPE", "POG_DIVISION_CODE", "POG_DEPT_CODE", "SUBDEPT",
                                                                                                                            "ADJUSTED_SUBDEPT_METERAGE", "Year", "Month", "Is_Holiday", "CPI"]], df_mso_usage_week_subdept_all_hol_ym_cpi_ordenc[df_mso_usage_week_subdept_all_hol_ym_cpi_ordenc["LOC_KEY"] == 1333]["WEEKLY_SALES"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.4, random_state = 0) # Can adjust test_size

## Train AdaBoost for classification.
## Reference:
## https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html
from sklearn.ensemble import AdaBoostClassifier
#adaboost_classify = AdaBoostClassifier(n_estimators = 100, random_state = 0)

## Train AdaBoost for regression.
## Reference:
## https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html
## https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
adaboost_regr = AdaBoostRegressor(DecisionTreeRegressor(max_depth = 10), random_state = 0, n_estimators = 50)
adaboost_regr.fit(x_train, y_train)

# Output mean accuracy: R^2 (since it is a regression) with the same data features and labels.
# Reference:
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html#sklearn.ensemble.AdaBoostClassifier.score
# https://www.kaggle.com/getting-started/27261
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html#sklearn.metrics.r2_score
# https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation
st.write("AdaBoost mean accuracy:")
st.write(adaboost_regr.score(x_test, y_test)) # For best model, R^2 = 1.0)

# Plot of mean decrease in impurity against features.
# Reference:
# https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html
import time
import numpy as np
import pandas as pd

## Compute standard deviation of accumulation of impurity decrease within each tree for each feature.
std = np.std([tree.feature_importances_ for tree in adaboost_regr.estimators_], axis = 0)

# Sort feature importances by descending order.
# Reference:
# https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.sort_values.html
x_feature_names = x.columns.values
adaboost_regr_importances = pd.Series(adaboost_regr.feature_importances_, index = x_feature_names)
adaboost_regr_importances = adaboost_regr_importances.sort_values(ascending = False)

df_adaboost_regr_importances = pd.DataFrame(adaboost_regr_importances, columns = ["Gini Importances"])
st.write("Feature Importances:")
st.table(df_adaboost_regr_importances)

# Plot of mean decrease in impurity against features.
# Reference:
# https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html
import time
import numpy as np
import pandas as pd

## Compute standard deviation of accumulation of impurity decrease within each tree for each feature.
std = np.std([tree.feature_importances_ for tree in adaboost_regr.estimators_], axis = 0)

## Plot results.
## Reference:
## https://matplotlib.org/devdocs/gallery/subplots_axes_and_figures/figure_size_units.html
fig, ax = plt.subplots(figsize = (15, 7))
adaboost_regr_importances.plot.bar(yerr = std, ax = ax)
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()

st.write("Plot of Feature Importances:")
st.pyplot(fig)

# Prediction of "WEEKLY_SALES" by unique "WK_IDNT" of store's location: 1333 using 100 decision trees.
# Reference:
# https://scikit-learn.org/stable/auto_examples/ensemble/plot_adaboost_regression.html#decision-tree-regression-with-adaboost
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor

x_L1333_sales_margins = pd.DataFrame(df_mso_usage_week_subdept_all_hol_ym_cpi_L1333_sales_margins_gb_week.index, columns = ["WK_IDNT"])
y_L1333_sales_margins = df_mso_usage_week_subdept_all_hol_ym_cpi_L1333_sales_margins_gb_week["WEEKLY_SALES"]
rng = np.random.RandomState(1)

# Train decision tree.
regr_1 = DecisionTreeRegressor(max_depth = 4)
regr_1.fit(x_L1333_sales_margins, y_L1333_sales_margins)

# Train adaboost.
regr_2 = AdaBoostRegressor(DecisionTreeRegressor(max_depth = 10),
                           n_estimators = 100, random_state = rng)
regr_2.fit(x_L1333_sales_margins, y_L1333_sales_margins)

# Make predictions.
y_1 = regr_1.predict(x_L1333_sales_margins)
y_2 = regr_2.predict(x_L1333_sales_margins)

# Reference:
# https://matplotlib.org/stable/gallery/subplots_axes_and_figures/subplots_demo.html
fig1, ax1 = plt.subplots()
ax1.scatter(x_L1333_sales_margins, y_L1333_sales_margins, c = "k", label = "training samples")
ax1.plot(x_L1333_sales_margins, y_1, c = "g", label = "n_estimators = 1", linewidth = 2)
ax1.plot(x_L1333_sales_margins, y_2, c = "r", label = "n_estimators = 100", linewidth = 2)
ax1.set(xlabel = "data", ylabel = "target")
ax1.legend()

st.write("Boosted Decision Tree Regression:")
st.pyplot(fig1)
'''
AdaBoost mean accuracy (by Weeks):
'''
st.write(regr_2.score(x_L1333_sales_margins, y_L1333_sales_margins))












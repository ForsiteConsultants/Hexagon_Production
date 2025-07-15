import os
import string
from collections import deque
import pickle
# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import altair as alt
import math

# data
from sklearn import datasets
from sklearn.compose import ColumnTransformer, make_column_transformer

# Classifiers
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.impute import SimpleImputer

# classifiers / models
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, RobustScaler


from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression

# other
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    cross_val_score,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import r2_score

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import BayesianRidge
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from functions_pkgs import *

def scatter_plts(data, plt_title, x_column, y_column, x_title, y_title, axis_max,
                 add_rmse = False, rmse_ = 1, rmse_p=1):
    """
    data: data frame
    plt_title: str type
    x_column: str type, the column name
    x_title: str tytpe, x axis label
    axis_limit: the maximum number
    """

    plt1 = alt.Chart(data).mark_point().encode(
    x=alt.X(x_column, axis=alt.Axis(title=x_title)),
    y=alt.Y(y_column, axis=alt.Axis(title=y_title))
    ).properties(
    title = plt_title
    )

    abline = alt.Chart(pd.DataFrame({
        x_column:[0, axis_max],
        y_column: [0, axis_max]})).mark_line(
            color = 'red'
        ).encode(
            x = x_column,
            y = y_column
        )

    final_plt = plt1 + abline

    text = alt.Chart({'values':[{}]}).mark_text(align="left", baseline="top").encode(
        x=alt.value(5),  # pixels from left
        y=alt.value(5),  # pixels from top
        text=alt.value([f"RMSE: {rmse_:.2f}", f"RMSE%: {rmse_p:.1f}"]))

    # text = alt.Chart({'values':[{}]}).mark_text(align="left", baseline="top").encode(
    #     x=alt.value(5),  # pixels from left
    #     y=alt.value(5),  # pixels from top
    #     text=alt.value([f"RMSE: {rmse_:.2f}"]))
    
    box = alt.Chart({'values':[{}]}).mark_rect(stroke='black', color='white').encode(
        x=alt.value(3),
        x2=alt.value(80),
        y=alt.value(3),
        y2=alt.value(30))
    final_plt_text = plt1 + abline + box + text

    if add_rmse:
        return final_plt_text
    else:
        return final_plt


def mean_std_cross_val_scores(model, X_train, y_train, **kwargs):
    """
    Returns mean and std of cross validation.
    """
    scores = cross_validate(model, X_train, y_train, **kwargs)

    mean_scores = pd.DataFrame(scores).mean()
    std_scores = pd.DataFrame(scores).std()
    out_col = []

    for i in range(len(mean_scores)):
        out_col.append((f"%0.3f (+/- %0.3f)" % (abs(mean_scores[i]), std_scores[i])))

    return pd.Series(data=out_col, index=mean_scores.index)


def select_model(var, data_root, regressors, scoring_metric="neg_root_mean_squared_error", plotid='PLOT_ID'):
    
    # define model varibales
    model_name = var[5:]
    plot_var = var
    iti_var = model_name

    # load in data
    input_df = pd.read_csv(os.path.join(data_root, var + '_clean_data.csv')).set_index(plotid)
    varx_df = pd.read_csv(os.path.join(data_root, var + '_new_VARX.csv'))
    varx_list = varx_df[varx_df['IO'] == 'X'].varName.tolist()
    varx_list.append(var)

    # check if ITI variable in the varx list
    if var[5:] in varx_list:
        pass
    else:
        varx_list.append(var[5:])
    df_cln = input_df[varx_list]
    # print(df_cln[[var]])
    # print('Number of sample plots and columns are ', df_cln.shape)


    # define plot attributes
    # plot axis
    axis_max = math.ceil(np.max(df_cln[[var, iti_var]].values)/10)*10
    # plot 1 initial ITI to Field comparison plot
    plt_title1 = model_name
    x_axis1 = 'ITI ' + model_name 
    y_axis1 = 'Field ' + model_name

    
    plt1 = scatter_plts(df_cln, 
                    plt_title1, 
                    iti_var, 
                    plot_var,
                    x_axis1, 
                    y_axis1, 
                    axis_max)
    

    q_95 = df_cln[plot_var].quantile(q=0.95)
    max_y = df_cln[plot_var].max()
    # print(q_95, max_y)

    # FIT SIMPLE LINEAR REGRESSION
    x_input = df_cln[[iti_var]]
    y_input = df_cln[plot_var]
    reg = LinearRegression(fit_intercept=False).fit(x_input, y_input)
    b1 = reg.coef_[0]
    # print('Coefficients for linear model are ', round(b1, 2))

    # set columns
    columns = list(df_cln.columns.values)
    num_features = [i for i in columns if not i.startswith('PLOT')]
    quant_bins = pd.qcut(df_cln[plot_var], q=5, labels=False, duplicates='drop')

    # split data
    train_df, test_df = train_test_split(df_cln, test_size = 0.2, random_state=123, stratify=quant_bins)
    X_train, y_train = train_df[num_features], train_df[plot_var]
    X_test, y_test = test_df[num_features], test_df[plot_var]

    # check if theres any null values
    if train_df.isnull().values.any():
        print('data has null values, CHECK!')
    else:
        pass
    
    results = {}
    # Set the baseline scenario for comparison
    dummy = DummyRegressor(strategy="mean")
    results["Dummy"] = mean_std_cross_val_scores(dummy, X_train, y_train, return_train_score=True, scoring=scoring_metric)
    

    for (name, model) in regressors.items():
        results[name] = mean_std_cross_val_scores(
            model, X_train, y_train, return_train_score=True, scoring=scoring_metric
        )

    return axis_max, q_95, max_y, b1, df_cln, plt1, results
    
        

def run_model_general(var, scaler, df_cln, pipeline, axis_max, scoring_metric='neg_root_mean_squared_error', grid_search=False):
    """
    this function use pipeline from general pool, instead of only elastic net
    Input:
    var: variable name starts with PLOT. i.e. PLOT_CON_GVOL_PRED_HA
    scaler: usually it is the standard scaler to use
    df_cln: cleaned
    
    """
    model_name = iti_var = var[5:]
    plot_var = var
    pred_var = 'PRED_' + model_name
    adj_var = 'ADJ_' + model_name
    
    # set columns
    columns = list(df_cln.columns.values)
    num_features = [i for i in columns if not i.startswith('PLOT')]
    quant_bins = pd.qcut(df_cln[plot_var], q=5, labels=False, duplicates='drop')
    # split data
    train_df, test_df = train_test_split(df_cln, test_size = 0.2, random_state=123, stratify=quant_bins)
    X_train, y_train = train_df[num_features], train_df[plot_var]
    X_test, y_test = test_df[num_features], test_df[plot_var]
    
    # plot 2 training plots - predicted vs field
    plt_title2 = model_name + ' Training'
    x_axis2 = 'Predicted ' + model_name
    y_axis2 = 'Field ' + model_name
    plt_title2_2 = model_name + ' Training (ADJUSTED)'

    # plot 3 validation plots - predicted vs field
    plt_title3 = model_name + ' Validation'
    x_axis3 = 'Predicted ' + model_name
    y_axis3 = 'Field ' + model_name
    plt_title3_3 = model_name + ' Validation (ADJUSTED)'

    # define the grid search
    if grid_search == True:
        param_grid = {
            "elasticnet__alpha": 10.0 ** np.arange(-3, 3),
            "elasticnet__l1_ratio": np.arange(0, 1, 0.01)
        }
        grid_search = GridSearchCV(pipeline, param_grid, scoring=scoring_metric,
        cv = 5, n_jobs = -1, return_train_score = True)
        grid_search.fit(X_train, y_train)
        print('----------------------------------------')
        print("Best cv score from grid search: %.3f" % grid_search.best_score_)
        grid_search.best_params_
        p_alpha = grid_search.best_params_['elasticnet__alpha']
        p_l1_ratio = grid_search.best_params_['elasticnet__l1_ratio']
        # define the final pipeline
        pipe_final = make_pipeline(scaler, ElasticNet(alpha=p_alpha, l1_ratio=p_l1_ratio, random_state=2))
    else:
        pipe_final = pipeline
    pipe_final.fit(X_train, y_train)
    y_pred = pipe_final.predict(X_train)

    result_df = train_df[[plot_var, iti_var]].copy()
    result_df[pred_var] = y_pred
    result_df['DIFF'] = result_df[plot_var] - result_df[pred_var]
    # model the error
    x_input = result_df[[pred_var]]
    y_input = result_df.DIFF
    reg = LinearRegression().fit(x_input, y_input)
    b0_error = reg.intercept_
    b1_error = reg.coef_[0]
    # print('Coefficients for linear model are ', round(b0_error, 2), ' and ', round(b1_error, 2))

    diff_pred = reg.predict(x_input)
    result_df['DIFF_PRED'] = diff_pred
    result_df[adj_var] = result_df[pred_var] + result_df['DIFF_PRED']

    # calculate rmse
    rmse_ = root_mean_squared_error(result_df[plot_var], result_df[pred_var])
    rmse_p = rmse_/df_cln[plot_var].mean()*100
    adj_rmse_ = root_mean_squared_error(result_df[plot_var], result_df[adj_var])
    adj_rmse_p = adj_rmse_/df_cln[plot_var].mean()*100

    # training plots
    plt2 = scatter_plts(result_df, plt_title2, 
                        pred_var, plot_var, 
                        x_axis2,
                        y_axis2,
                        axis_max, True, rmse_, rmse_p)
    plt2_2 = scatter_plts(result_df, plt_title2_2, 
                        adj_var, plot_var, 
                        x_axis2,
                        y_axis2,
                        axis_max, True, adj_rmse_, adj_rmse_p)
    combined_plot1 = alt.hconcat(plt2, plt2_2)

    # calculate predictions
    y_pred = pipe_final.predict(X_test)
    result_df_test = test_df[[plot_var, iti_var]].copy()
    result_df_test[pred_var] = y_pred
    result_df_test[adj_var] = b0_error + (1 + b1_error) *result_df_test[pred_var]
    # calculate rmse
    rmse_ = root_mean_squared_error(result_df_test[plot_var], result_df_test[pred_var])
    rmse_p = rmse_/df_cln[plot_var].mean()*100
    adj_rmse_ = root_mean_squared_error(result_df_test[plot_var], result_df_test[adj_var])
    adj_rmse_p = adj_rmse_/df_cln[plot_var].mean()*100

    # validation plots
    plt3 = scatter_plts(result_df_test, plt_title3,
                        pred_var, 
                        plot_var,
                        x_axis3,
                        y_axis3,
                        axis_max, True, rmse_, rmse_p)
    plt3_3 = scatter_plts(result_df_test, plt_title3_3,
                        adj_var, 
                        plot_var,
                        x_axis3,
                        y_axis3,
                        axis_max, True, adj_rmse_, adj_rmse_p)
    combined_plot2 = alt.hconcat(plt3, plt3_3)

    return pipe_final, b0_error, b1_error, combined_plot1, combined_plot2

def run_model_elasticNet(var, scaler, df_cln, pipeline, axis_max, scoring_metric='neg_root_mean_squared_error'):

    model_name = iti_var = var[5:]
    plot_var = var
    pred_var = 'PRED_' + model_name
    adj_var = 'ADJ_' + model_name
    
    # set columns
    columns = list(df_cln.columns.values)
    num_features = [i for i in columns if not i.startswith('PLOT')]
    quant_bins = pd.qcut(df_cln[plot_var], q=5, labels=False, duplicates='drop')
    # split data
    train_df, test_df = train_test_split(df_cln, test_size = 0.2, random_state=123, stratify=quant_bins)
    X_train, y_train = train_df[num_features], train_df[plot_var]
    X_test, y_test = test_df[num_features], test_df[plot_var]
    
    # plot 2 training plots - predicted vs field
    plt_title2 = model_name + ' Training'
    x_axis2 = 'Predicted ' + model_name
    y_axis2 = 'Field ' + model_name
    plt_title2_2 = model_name + ' Training (ADJUSTED)'

    # plot 3 validation plots - predicted vs field
    plt_title3 = model_name + ' Validation'
    x_axis3 = 'Predicted ' + model_name
    y_axis3 = 'Field ' + model_name
    plt_title3_3 = model_name + ' Validation (ADJUSTED)'

    # define the grid search
    param_grid = {
        "elasticnet__alpha": 10.0 ** np.arange(-3, 3),
        "elasticnet__l1_ratio": np.arange(0, 1, 0.01)
    }
    grid_search = GridSearchCV(pipeline, param_grid, scoring=scoring_metric,
    cv = 5, n_jobs = -1, return_train_score = True)
    grid_search.fit(X_train, y_train)
    print('----------------------------------------')
    print("Best cv score from grid search: %.3f" % grid_search.best_score_)
    grid_search.best_params_
    p_alpha = grid_search.best_params_['elasticnet__alpha']
    p_l1_ratio = grid_search.best_params_['elasticnet__l1_ratio']
    # define the final pipeline
    pipe_final = make_pipeline(scaler, ElasticNet(alpha=p_alpha, l1_ratio=p_l1_ratio, random_state=2))
    pipe_final.fit(X_train, y_train)
    y_pred = pipe_final.predict(X_train)

    result_df = train_df[[plot_var, iti_var]].copy()
    result_df[pred_var] = y_pred
    result_df['DIFF'] = result_df[plot_var] - result_df[pred_var]
    # model the error
    x_input = result_df[[pred_var]]
    y_input = result_df.DIFF
    reg = LinearRegression().fit(x_input, y_input)
    b0_error = reg.intercept_
    b1_error = reg.coef_[0]
    # print('Coefficients for linear model are ', round(b0_error, 2), ' and ', round(b1_error, 2))

    diff_pred = reg.predict(x_input)
    result_df['DIFF_PRED'] = diff_pred
    result_df[adj_var] = result_df[pred_var] + result_df['DIFF_PRED']

    # calculate rmse
    rmse_ = root_mean_squared_error(result_df[plot_var], result_df[pred_var])
    rmse_p = rmse_/df_cln[plot_var].mean()*100
    adj_rmse_ = root_mean_squared_error(result_df[plot_var], result_df[adj_var])
    adj_rmse_p = adj_rmse_/df_cln[plot_var].mean()*100

    # training plots
    plt2 = scatter_plts(result_df, plt_title2, 
                        pred_var, plot_var, 
                        x_axis2,
                        y_axis2,
                        axis_max, True, rmse_, rmse_p)
    plt2_2 = scatter_plts(result_df, plt_title2_2, 
                        adj_var, plot_var, 
                        x_axis2,
                        y_axis2,
                        axis_max, True, adj_rmse_, adj_rmse_p)
    combined_plot1 = alt.hconcat(plt2, plt2_2)

    # calculate predictions
    y_pred = pipe_final.predict(X_test)
    result_df_test = test_df[[plot_var, iti_var]].copy()
    result_df_test[pred_var] = y_pred
    result_df_test[adj_var] = b0_error + (1 + b1_error) *result_df_test[pred_var]
    # calculate rmse
    rmse_ = root_mean_squared_error(result_df_test[plot_var], result_df_test[pred_var])
    rmse_p = rmse_/df_cln[plot_var].mean()*100
    adj_rmse_ = root_mean_squared_error(result_df_test[plot_var], result_df_test[adj_var])
    adj_rmse_p = adj_rmse_/df_cln[plot_var].mean()*100

    # validation plots
    plt3 = scatter_plts(result_df_test, plt_title3,
                        pred_var, 
                        plot_var,
                        x_axis3,
                        y_axis3,
                        axis_max, True, rmse_, rmse_p)
    plt3_3 = scatter_plts(result_df_test, plt_title3_3,
                        adj_var, 
                        plot_var,
                        x_axis3,
                        y_axis3,
                        axis_max, True, adj_rmse_, adj_rmse_p)
    combined_plot2 = alt.hconcat(plt3, plt3_3)

    return pipe_final, b0_error, b1_error, combined_plot1, combined_plot2


def save_model_output(var, output_root, pipe_final, values_list, df_cln, plotid = 'PLOT_ID'):
    """
    input:
    var: variable name, it is the 'PLOT_' + base name. i.e. PLOT_BPH
    output_root: the folder path to store the model output
    pipe_final: the final pipeline decided to use
    values_list: a list to store key information - model_name, q_95, max_y, b1, b0_error, b1_error
    df_cln: clean data frame has X and Y variables
    
    return:
    dump the output as pickly file
    save the model information as csv
    save the feature importance as csv
    """

    model_name = var[5:]
    plot_var = var
    iti_var = model_name
    pred_var = 'PRED_' + model_name
    final_output = model_name + '.sav'
    filename = os.path.join(output_root, final_output)
    pickle.dump(pipe_final, open(filename, 'wb'))

    # model info csv
    df = pd.DataFrame(columns =['MODEL_NAME', 'Q95', 'MAX', 'B1', 'B0_ERROR', 'B1_ERROR'], data=[values_list])
    df.to_csv(os.path.join(output_root, model_name + '_Model_Info.csv'))
    # print(df)

    # save predicted data
    # save out plot data predicted value using final model
    b0 = values_list[4]
    b1 = values_list[5]
    loaded_model = pickle.load(open(filename, 'rb'))
    columns = list(df_cln.columns.values)
    num_features = [i for i in columns if not i.startswith('PLOT')]
    X_all = df_cln[num_features]
    y_pred = loaded_model.predict(X_all)

    # copy of plot data only have PLOTID and veriables
    df_copy = df_cln[[plot_var, iti_var]].copy()
    df_copy[pred_var] = b0 + b1*y_pred + y_pred
    # save output to csv
    df_output = os.path.join(output_root, model_name + '_plot_output.csv')
    df_copy.to_csv(df_output)

    # model importance csv
    output = os.path.join(output_root, model_name + '_feature_importance.csv')
    regressor_name = pipe_final.steps[-1][1].__class__.__name__
    if regressor_name == 'RandomForestRegressor':
        model_coefs = pd.DataFrame(data=pipe_final[1].feature_importances_.flatten(), index=num_features, columns=["Coefficient"])
    else:
        model_coefs = pd.DataFrame(data=pipe_final[1].coef_.flatten(), index=num_features, columns=["Coefficient"])
    model_coefs['magnitude'] = model_coefs['Coefficient'].abs()
    fc_importance = model_coefs[model_coefs['magnitude'] > 0].sort_values(by="magnitude", ascending=False)
    print(f'Model Used: {regressor_name}')
    print(f'The Most Important Five Predictors for Variable {model_name}')
    print(fc_importance.head(5))
    fc_importance.to_csv(output)




import numpy as np
from sklearn.model_selection import KFold
from sklearn.base import clone
from sklearn.linear_model import LinearRegression

import numpy as np
from sklearn.model_selection import KFold
from sklearn.base import clone
from sklearn.linear_model import LinearRegression

class SuperLearner_v1:
    def __init__(self, base_models, meta_model, n_folds=5):
        """
        Parameters:
            base_models (list): List of (name, model) tuples. Models can be pipelines.
            meta_model (object): Model instance for meta-learning.
            n_folds (int): Number of folds for cross-validation.
        """
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
        # Dictionary to store fitted models for each base model.
        self.fitted_base_models = {name: [] for name, _ in base_models}
    
    def fit(self, X, y):
        """
        Fit the Super Learner model.
        
        Parameters:
            X (DataFrame or numpy array): Training features.
            y (Series or numpy array): Target variable.
        """
        n_samples = X.shape[0]
        n_models = len(self.base_models)
        # Create an array to store out-of-fold predictions for meta-model training.
        meta_features = np.zeros((n_samples, n_models))
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        
        for model_idx, (name, model) in enumerate(self.base_models):
            # Clear any previously stored models.
            self.fitted_base_models[name] = []
            
            for train_idx, valid_idx in kf.split(X, y):
                # Use .iloc for DataFrame indexing; if X is a numpy array, normal indexing works.
                try:
                    X_train_fold, X_valid_fold = X.iloc[train_idx], X.iloc[valid_idx]
                except AttributeError:
                    X_train_fold, X_valid_fold = X[train_idx], X[valid_idx]
                
                try:
                    y_train_fold, y_valid_fold = y.iloc[train_idx], y.iloc[valid_idx]
                except AttributeError:
                    y_train_fold, y_valid_fold = y[train_idx], y[valid_idx]
                
                # Clone the model (works with pipelines as well).
                model_clone = clone(model)
                model_clone.fit(X_train_fold, y_train_fold)
                
                # Save the fitted clone for later predictions.
                self.fitted_base_models[name].append(model_clone)
                
                # Generate out-of-fold predictions for the meta-model.
                meta_features[valid_idx, model_idx] = model_clone.predict(X_valid_fold)
        
        # Fit the meta-model using the out-of-fold predictions.
        self.meta_model.fit(meta_features, y)
    
    def predict(self, X):
        """
        Make predictions with the fitted Super Learner.
        
        Parameters:
            X (DataFrame or numpy array): Features for which to predict.
        
        Returns:
            np.array: Final predictions.
        """
        n_samples = X.shape[0]
        n_models = len(self.base_models)
        meta_features = np.zeros((n_samples, n_models))
        
        # For each base model, average predictions over the fitted clones from each fold.
        for model_idx, (name, _) in enumerate(self.base_models):
            fold_preds = []
            for fitted_model in self.fitted_base_models[name]:
                fold_preds.append(fitted_model.predict(X))
            meta_features[:, model_idx] = np.mean(fold_preds, axis=0)
        
        # Use the meta-model to generate the final predictions.
        return self.meta_model.predict(meta_features)



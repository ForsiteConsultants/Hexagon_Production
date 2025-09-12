#
#
#  This is the key strip that takes the ITI data organized in grids and rolls-up
#  individual tree info into hexagon level sums, averages, and distribution info
#  for specific attributes. Hexagon attributes are dumped to temporary csv files
#  that are later written to the hexagon feature class. 
#   
#   Dataframes in the script are passed to the trees_to_csv.py script for processing
#   key references: 
#       grouped up plotchar files for additional attributes used in adjustments
#       ITI trees gdb - organized into processing groups and grids
#       hexagon feature class  
#  

import os
import time
import os
import string
from collections import deque
import pickle
# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from math import sqrt
# import altair as alt

# data
from sklearn import datasets
from sklearn.compose import ColumnTransformer, make_column_transformer

# Classifiers
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.impute import SimpleImputer
#mlens
import collections
import collections.abc
collections.Sequence = collections.abc.Sequence #monkey patch for mlens
from mlens.ensemble import SuperLearner
# classifiers / models
from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error
# quick fix for processing computer 1:
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

from sklearn.preprocessing import StandardScaler, MaxAbsScaler, RobustScaler

# other
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    cross_val_score,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, MinMaxScaler
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

from multiprocessing import Pool
import warnings
warnings.filterwarnings("ignore")
import yaml
from shared.logger_utils import get_logger

logger = get_logger('3003_c_MultiProcess_modeling_to_csv')

def root_mean_squared_error(y_true, y_pred):
    """Return RMSE, equivalent to sklearn ≥1.4 helper."""
    return mean_squared_error(y_true, y_pred, squared=False)

def rmse(yreal, yhat):
	return sqrt(mean_squared_error(yreal, yhat))

def read_yaml_config():
	"""
	Read yaml config and return dictionary of items
	"""
	yml_file = r'S:\1845\5\03_MappingAnalysisData\03_Scripts\06_HexProduction\config_hex_G.yml'
	with open(yml_file, 'r') as file:
		config = yaml.safe_load(file)
		return config

# original folder including ITI, some CSV output, model output
# define working directory
config = read_yaml_config()
hex_root = config['root_folder']
hex_output_folder = config['hex_output_folder']
hex_gdb = config['hex_gdb']
hex_fc = config['hex_fc']
hex_output = os.path.join(hex_output_folder, hex_gdb, hex_fc)
area = 'AREA_G'
hex_y = config['hex_y_variables']
treelist_y = config['treelist_y_variables']
compiled_grids_folder = config['compiled_grids_folder']
csv_folder = config['csv_folder']
plot_chars_folder = config['plot_chars_folder']
boruta_output_folder = config['boruta_output_folder']
model_output_folder = config['model_output_folder']


# merge model input data
def hex_modeling_output(grid, compiled_grids_folder, csv_folder, plot_chars_folder, hexid='HEXID'):
    iti_comp = pd.read_csv(os.path.join(compiled_grids_folder, grid, 'ITI_compile_' + grid + '.csv'), low_memory=False).set_index(hexid)
    plot_chars = pd.read_csv(os.path.join(plot_chars_folder, grid + '_PlotChars_bil.csv'), low_memory=False).set_index(hexid)
    # climate = pd.read_csv(os.path.join(csv_folder, grid, grid + '_climate_data.csv'), low_memory=False).set_index(hexid)
    # df = iti_comp.join(plot_chars, how='inner').join(climate, how='inner')
    df = iti_comp.join(plot_chars, how='inner')

    iti_ranme_map= {'TPMD': 'TPM_D',
                    'GVPHD': 'GVPH_D',} 
    for col in iti_ranme_map.keys():
        if col in df.columns:
            df.rename(columns={col: iti_ranme_map[col]}, inplace=True)

    # # check the number of rows of each dataframe
    # indices_diff1 = iti_comp.index.difference(plot_chars.index).tolist()
    # indices_diff2 = iti_comp.index.difference(climate.index).tolist()
    # if len(indices_diff1) > 0 or len(indices_diff2) > 0:
    #     print('number of rows are different! ', grid)
    #     print('differences iti vs plot chars:', len(indices_diff1))
    #     print('differences iti vs climate:', len(indices_diff2))

    # some data cleaning
    df = df.select_dtypes(exclude=['object', 'category'])
    df = df[[col for col in df.columns if 'class' not in col]]
    df = df[[col for col in df.columns if '_pct' not in col]]
    df.columns = df.columns.str.replace(' ', '') ## remove white space of column names, drives me crazy

    prefix = ''
    df[prefix + 'CON_GVOL_PRED_HA'] = df['GVPH_con'] 
    df[prefix + 'DEC_GVOL_PRED_HA'] = df['GVPH_dec'] 
    df[prefix + 'CON_GMVOL_PRED_HA'] = df['MVPH_con'] 
    df[prefix + 'DEC_GMVOL_PRED_HA'] = df['MVPH_dec']
    df[prefix + 'CON_SPH_GT_5m'] = df['SPH_con']
    df[prefix + 'DEC_SPH_GT_5m'] = df['SPH_dec']
    df[prefix + 'CON_MERCH_SPH'] = df['MSPH_con'] 
    df[prefix + 'DEC_MERCH_SPH'] = df['MSPH_dec']
    df[prefix + 'CON_STEM_PER_M3'] = df['MTPM_con'] 
    df[prefix + 'DEC_STEM_PER_M3'] = df['MTPM_dec'] 
    df[prefix + 'CON_BA_HA'] = df['BPH_con']
    df[prefix + 'DEC_BA_HA'] = df['BPH_dec']
    df[prefix + 'CON_MBA_HA'] = df['MBPH_con'] 
    df[prefix + 'DEC_MBA_HA'] = df['MBPH_dec']
    df[prefix + 'CON_QMD'] = np.where(df[prefix + 'CON_SPH_GT_5m'] > 0, np.sqrt(df[prefix + 'CON_BA_HA' ]/(df[prefix + 'CON_SPH_GT_5m'] * 0.0000785)), 0)
    df[prefix + 'DEC_QMD'] = np.where(df[prefix + 'DEC_SPH_GT_5m'] > 0, np.sqrt(df[prefix + 'DEC_BA_HA' ]/(df[prefix + 'DEC_SPH_GT_5m'] * 0.0000785)), 0)


    for source in [ 'Hex', 'treeList']:
        print(f"    → processing source = {source!r}, grid = {grid!r}")
        df_combined = []
        if source == 'Hex':
            var_list = hex_y
        else:
            var_list = treelist_y
        for var in var_list:
            model = var[5:]
            pred_var = 'PRED_' + model
            varx_df = pd.read_csv(os.path.join(boruta_output_folder, var + '_new_VARX.csv'))
            varx_list = varx_df[(varx_df['IO'] == 'X') & (~varx_df['varName'].str.startswith('G_'))].varName.tolist()
            if model in varx_list:
                pass
            else:
                varx_list.append(var[5:])
            try:
                df_cln = df[varx_list]
                # check for NAN values - the original data contains nan
                na_cols = df_cln.columns[df_cln.isna().any()].tolist()
                if len(na_cols) > 0:
                    print(' data contains nan! check columns: ', grid, na_cols)
                    df_cln = df_cln.dropna(axis=0, how='any') # drop nan rows due to differences on input data
            except Exception as e:
                print(f"❗  ERROR  grid={grid!r}  source={source!r}  var={var!r}  missing cols: {e}")
                continue
                # print(grid, var, e)
                # continue

            model_info = pd.read_csv(os.path.join(model_output_folder, source, model + '_Model_Info.csv'))
            b0 = model_info.at[0, 'B0_ERROR']
            b1 = model_info.at[0, 'B1_ERROR']
            if 'CON' in model:
                max_limit = model_info.at[0, 'MAX']*1.2
            else:
                max_limit = model_info.at[0, 'MAX']*2

            if source == 'Hex':
                model_output = os.path.join(model_output_folder, source, model + '_superLearner.pkl') 
            else:
                model_output = os.path.join(model_output_folder, source, model + '.sav')

            # Confirm modl file exist
            if not os.path.exists(model_output):
                raise FileNotFoundError(f"Model file does not exist: {model_output}")
            if os.path.getsize(model_output) == 0:
                raise ValueError(f"Model file is empty: {model_output}")
            
            try:
                with open(model_output, 'rb') as f:
                    loaded_model = pickle.load(f)
            except Exception as e:
                print(f"Failed to load model {model_output}")
                raise e
            
            #loaded_model = pickle.load(open(model_output, 'rb'))
            columns = list(df_cln.columns.values)
            X_all = df_cln[columns]
            try:
                y_pred = loaded_model.predict(X_all)
            except Exception as e:
                print(grid, var)
                continue
            
            df_copy = df_cln[[model]].copy()
            df_copy[pred_var] = b0 + b1*y_pred + y_pred
            df_copy[pred_var] = np.where(df_copy[pred_var] < 0, df_copy[model], df_copy[pred_var])
            df_copy[pred_var] = np.where(df_copy[pred_var] > max_limit, max_limit, df_copy[pred_var])
            df_copy = df_copy.drop([model], axis=1)

            df_combined.append(df_copy)

        df_final = pd.concat(df_combined, axis=1)
        if source == 'Hex':
            try:
                df_con = df_final[
                    ['PRED_CON_SPH_GT_5m', 'PRED_CON_BA_HA',
                     'PRED_DEC_SPH_GT_5m', 'PRED_DEC_BA_HA']
                ]
            except KeyError as e:
                print(f"❗  ERROR in grid '{grid}' for source '{source}': missing columns {e}")
                raise
            df_con.columns = ['SPH_con', 'BPH_con', 'SPH_dec', 'BPH_dec']
        if source == 'treeList':
            df_final.columns = [i[5:] for i in list(df_final.columns.values)]
            df_final = df_final.drop(columns=['SPH_con', 'BPH_con', 'SPH_dec', 'BPH_dec'])
            df_final = df_final.join(df_con, how='inner')

        print('writing csv for grid:', grid, source)

        out_dir = os.path.join(csv_folder, grid)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        df_final.to_csv(os.path.join(out_dir, grid + '_' + source + '_predicted_output_v5.csv'))


###############
# update path
Start = time.time()

multiprocess = multiprocess = r'S:\1845\5\03_MappingAnalysisData\02_Data\06_Hexagon_Production\02_Process\csv_output\MultiProcessing_files_input_' + area + '.csv'
grid_list = pd.read_csv(multiprocess).GRID.unique()
hexid = 'HEXID' ### need to check HEXID spelling !!!!!!!!!!!
grid_list.sort()

########################
##### test function
# grid = 'A16'
# hex_modeling_output(grid, compiled_grids_folder, csv_folder, plot_chars_folder)


######################
##### multi processing
args = [(grid, compiled_grids_folder, csv_folder, plot_chars_folder) for grid in grid_list]
if __name__ == '__main__':
    with Pool(processes=8) as pool:
        pool.starmap(hex_modeling_output, args)

End = time.time()

print(round((End - Start)/60, 2), ' mins to finish')
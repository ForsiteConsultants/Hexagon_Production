# this script combines CHM tiffs from various sources
# so that the final GOOD one
# that covers the whole AOI
# and no wired null values

import arcpy
import os
import pandas as pd


# ####################################################
import time
from numpy.lib.type_check import mintypecode
import numpy as np
from shared.trees_to_csv_sp_split_ami import *



config = read_yaml_config()
hex_root = config['root_folder']
hex_output_folder = config['hex_output_folder']
hex_orig_folder = config['hex_orig_folder']
hex_gdb = config['hex_gdb']
hex_fc = config['hex_fc']
hex_output = os.path.join(hex_orig_folder, hex_gdb, hex_fc + '_merged')
hexid = 'HEXID'
csv_folder = config['csv_folder']

# ######################
# #######################

# grids to be processed:
df = pd.read_csv(os.path.join(csv_folder, 'MultiProcessing_files_input_AREA_G.csv'))
grid_list = df.GRID.tolist()
grid_list.sort()
# grid_list = ['AA_12', 'AA_13']
# grid = 'AA_2'
grid_list = sorted([x for x in grid_list if x not in [0, '0']])

Start = time.time()
arcpy.env.workspace = os.path.join(hex_orig_folder, hex_gdb)
arcpy.env.overwriteOutput = True
print('Started')
hex_grid_folder = os.path.join(hex_output_folder, 'GRID')
fc_grid = [os.path.join(hex_grid_folder, 'HEX_' + grid + '.gdb', grid) for grid in grid_list]
arcpy.Merge_management(fc_grid, hex_output)
arcpy.AlterField_management(hex_output, 'HEXID', 'HEX_ID', 'HEX_ID')
End = time.time()

print(f"it takes {round((End - Start)/60, 2)} minutes to finish the script")
#####################################


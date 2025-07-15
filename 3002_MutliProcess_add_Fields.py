import arcpy
from arcpy.ddd import Buffer3D
import pandas as pd
import numpy as np
import math, os, shutil
import time
from trees_to_csv_sp_split_ami import *
from multiprocessing import Pool

#Adding fields to the hexagon feature classes to prepare for summarized data
Start = time.time()
# define working directory
config = read_yaml_config()
hex_root = config['root_folder']
hex_orig_folder = config['hex_orig_folder']
hex_output_folder = config['hex_output_folder']
hex_gdb = config['hex_gdb']
hex_fc = config['hex_fc']
sr = config['spatial_reference']
hex_orig = os.path.join(hex_orig_folder, hex_gdb, hex_fc)
hex_output = os.path.join(hex_output_folder, hex_gdb, hex_fc)
area = 'AREA_B'


########################################
### Data Prep for multi processing

def multi_add_fields(hex_shp_folder, grid, output_folder, field_list, fields_dict):
    """
    hex_shp_folder: directory for original shapefile
    grid: 4km grid name
    output folder: directory for output gdb by 4km grids
    field_list: fields need to be added
    """
    shp = os.path.join(hex_shp_folder, 'GRID_' + grid, 'GRID_' + grid + '.shp')
    gdb_path = os.path.join(output_folder, 'HEX_' + grid + '.gdb')
    if os.path.exists(gdb_path):
        pass
    else:
        arcpy.CreateFileGDB_management(output_folder, 'HEX_' + grid + '.gdb')
        # copy shapefile into gdb
        out_fc = os.path.join(gdb_path, grid)
        arcpy.CopyFeatures_management(shp, out_fc)
        print('start adding fields ', grid)
        for field in field_list:
            if field in ['HEX_ID', 'EXPTGRIDID', 'PRODGRIDID']:
                pass
            else:
                arcpy.AddField_management(out_fc, field, fields_dict[field]['FieldType'], '', '', fields_dict[field]['FieldLength'])


Start = time.time()

hex_shp_folder = r'S:\1845\2\03_MappingAnalysisData\02_Data\06_Hexagon_Production\HEX_ORIG\GRID_HP_unzipped'
output_folder = r'S:\1845\2\03_MappingAnalysisData\02_Data\06_Hexagon_Production\02_Process\hex_output\GRID'

# get the list of grids to process
multiprocess = r'S:\1845\2\03_MappingAnalysisData\02_Data\06_Hexagon_Production\02_Process\csv_output\MultiProcessing_files_input_' + area + '.csv'
grid_list = pd.read_csv(multiprocess).GRID.unique()
# grid_list = grid_list[0:2]
grid_list.sort()

# read in field template
fields = pd.read_excel(os.path.join(hex_root, 'Hex_Inventory_Proposed_Database_Deliverable.xlsx'), sheet_name='FieldListTemplate')
fields_dict = fields.set_index('FieldName').to_dict(orient='index')
field_list = fields.FieldName.tolist()

# grid = 'AA_2'

args = [(hex_shp_folder, grid, output_folder, field_list, fields_dict) for grid in grid_list]

if __name__ == '__main__':
    with Pool(processes=10) as pool:
        pool.starmap(multi_add_fields, args)

# multi_add_fields(hex_shp_folder, grid, output_folder, field_list, fields_dict)

End = time.time()

print(round((End - Start)/60, 2), ' mins to finish')
    

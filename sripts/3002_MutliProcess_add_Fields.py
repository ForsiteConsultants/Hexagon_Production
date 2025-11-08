import arcpy
from arcpy.ddd import Buffer3D
import pandas as pd
import numpy as np
import math, os, shutil
import time
from shared.trees_to_csv_sp_split_ami import *
from shared.logger_utils import get_logger
from multiprocessing import Pool

logger = get_logger('3002_MutliProcess_add_Fields')

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
area = 'AREA_G'


########################################
### Data Prep for multi processing


def multi_add_fields(hex_shp_folder, grid, output_folder, field_list, fields_dict):
    """
    hex_shp_folder: directory for original shapefile
    grid: grid name (e.g., 'A24')
    output_folder: directory for output GDB by grids
    field_list: list of field names to add
    fields_dict: dictionary with field properties (like FieldType and FieldLength)
    """
    try:
        shp = os.path.join(hex_shp_folder, 'GRID_' + grid, 'GRID_' + grid + '.shp')
        gdb_path = os.path.join(output_folder, 'HEX_' + grid + '.gdb')
        # If the file GDB doesn't exist, create it and copy the shapefile into it.
        if not os.path.exists(gdb_path):
            arcpy.CreateFileGDB_management(output_folder, 'HEX_' + grid + '.gdb')
            out_fc = os.path.join(gdb_path, grid)
            arcpy.CopyFeatures_management(shp, out_fc)
            logger.info(f"Created GDB and copied features for grid {grid}")
        else:
            # If the GDB exists, assume the feature class exists inside it.
            out_fc = os.path.join(gdb_path, grid)
            logger.info(f"GDB already exists for grid {grid}")
        logger.info(f"Start adding fields for grid {grid}")
        # Get a list of current field names (converted to lowercase for case-insensitive matching)
        existing_fields = [f.name.lower() for f in arcpy.ListFields(out_fc)]
        for field in field_list:
            # Skip certain predefined fields
            if field in ['HEX_ID', 'EXPTGRIDID', 'PRODGRIDID']:
                continue
            # Check if the field already exists; if so, skip adding it.
            if field.lower() in existing_fields:
                logger.info(f"Field {field} already exists in grid {grid}, skipping.")
            else:
                field_props = fields_dict[field]
                arcpy.AddField_management(out_fc, field, field_props['FieldType'], 
                                          field_precision="", field_scale="",
                                          field_length=field_props['FieldLength'])
                logger.info(f"Field {field} added for grid {grid}.")
    except Exception as e:
        logger.error(f"Failed processing grid {grid}: {e}", exc_info=True)



Start = time.time()

hex_shp_folder = r'S:\1845\5\03_MappingAnalysisData\02_Data\06_Hexagon_Production\01_HEX_GRID\unzipped'
output_folder = r'S:\1845\5\03_MappingAnalysisData\02_Data\06_Hexagon_Production\02_Process\hex_output\GRID'                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    

# get the list of grids to process
multiprocess = r'S:\1845\5\03_MappingAnalysisData\02_Data\06_Hexagon_Production\02_Process\csv_output\MultiProcessing_files_input_' + area + '.csv'


try:
    grid_list = pd.read_csv(multiprocess).GRID.unique()
    # grid_list = grid_list[0:2]
    grid_list.sort()
    logger.info(f"Loaded and sorted grid_list: {grid_list}")
except Exception as e:
    logger.error(f"Failed to load or sort grid list from {multiprocess}: {e}", exc_info=True)
    grid_list = []

# read in field template

try:
    fields = pd.read_excel(os.path.join(hex_root, 'Hex_Inventory_Proposed_Database_Deliverable.xlsx'), sheet_name='FieldListTemplate')
    fields_dict = fields.set_index('FieldName').to_dict(orient='index')
    field_list = fields.FieldName.tolist()
    logger.info(f"Loaded field template and field list: {field_list}")
except Exception as e:
    logger.error(f"Failed to load field template: {e}", exc_info=True)
    fields_dict = {}
    field_list = []


##########################
###### to test function
# grid = 'A16'
# multi_add_fields(hex_shp_folder, grid, output_folder, field_list, fields_dict)


#######################
#### MULTI PROCESSING
args = [(hex_shp_folder, grid, output_folder, field_list, fields_dict) for grid in grid_list]
cores = 8


if __name__ == '__main__':
    try:
        with Pool(processes=cores) as pool:
            pool.starmap(multi_add_fields, args)
        logger.info("Multi-processing completed successfully.")
    except Exception as e:
        logger.error(f"Multi-processing failed: {e}", exc_info=True)

    End = time.time()
    duration = round((End - Start)/60, 2)
    logger.info(f"Total time to finish: {duration} mins")
    

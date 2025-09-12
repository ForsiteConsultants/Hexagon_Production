import arcpy
import os
from shared.trees_to_csv_sp_split_ami import *
from shared.logger_utils import get_logger

logger = get_logger('1001_Data_Prep')

config = read_yaml_config()
hex_root = config['root_folder']
hex_orig_folder = config['hex_orig_folder']
hex_gdb = config['hex_gdb']
hex_fc = config['hex_fc']
area = config['area']
sr = config['spatial_reference']

hex_output = os.path.join(hex_orig_folder, hex_gdb, hex_fc)

# orignal hex shp by 4km grids
hex_shp = os.path.join(hex_root, '01_HEX_GRID','unzipped')


try:
    arcpy.CreateFileGDB_management(hex_orig_folder, hex_gdb)
    logger.info(f"Created FileGDB: {os.path.join(hex_orig_folder, hex_gdb)}")
except Exception as e:
    logger.error(f"Failed to create FileGDB: {e}", exc_info=True)
    print(f"Failed to create FileGDB: {e}")

arcpy.env.workspace = os.path.join(hex_orig_folder, hex_gdb)
arcpy.env.overwriteOutput = True
arcpy.env.outputCoordinateSystem = arcpy.SpatialReference(sr)

# merge all grid hex into one file

try:
    grids_list = os.listdir(hex_shp)
    logger.info(f"Loaded grids_list: {grids_list}")
    if 'GRID_HP' in grids_list:
        grids_list.remove('GRID_HP')
        logger.info("Removed 'GRID_HP' from grids_list. New list: %s", grids_list)
except Exception as e:
    logger.error(f"Failed to list grids in {hex_shp}: {e}", exc_info=True)
    print(f"Failed to list grids in {hex_shp}: {e}")
    grids_list = []

    

for i, grid in enumerate(grids_list):
    try:
        grid_fc = os.path.join(hex_shp, grid, grid + '.shp')
        lyr_name = "lyr_tmp"
        if arcpy.Exists(lyr_name):
            arcpy.Delete_management(lyr_name)
        arcpy.MakeFeatureLayer_management(grid_fc, 'lyr_tmp')
        if i == 0:
            arcpy.CopyFeatures_management('lyr_tmp', hex_output)
            logger.info(f"Copied features from {grid_fc} to {hex_output}")
        else:
            arcpy.Append_management('lyr_tmp', hex_output)
            logger.info(f"Appended features from {grid_fc} to {hex_output} (grid {i})")
    except Exception as e:
        logger.error(f"Failed processing grid {grid}: {e}", exc_info=True)
        print(f"Failed processing grid {grid}: {e}")

################################
# prep production data and create working gdb

work_gdb = os.path.join(hex_root, 'working.gdb')
try:
    if not arcpy.Exists(work_gdb):
        logger.info(f"Creating working gdb at {work_gdb}")
        arcpy.CreateFileGDB_management(hex_root, 'working.gdb')
    arcpy.env.workspace = work_gdb
    arcpy.env.overwriteOutput = True
    arcpy.env.outputCoordinateSystem = arcpy.SpatialReference(sr)
    output = os.path.join(work_gdb, area + '_EXPTGRID')
    grid_layer = os.path.join(hex_root, '04_Working', 'AreaG_ExportGrid.shp')
    arcpy.CopyFeatures_management(grid_layer, output)
    logger.info(f"Copied features from {grid_layer} to {output}")
except Exception as e:
    logger.error(f"Failed to prepare production data: {e}", exc_info=True)
    print(f"Failed to prepare production data: {e}")

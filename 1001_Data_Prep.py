import arcpy
import os
from trees_to_csv_sp_split_ami import *

config = read_yaml_config()
hex_root = config['root_folder']
hex_orig_folder = config['hex_orig_folder']
hex_gdb = config['hex_gdb']
hex_fc = config['hex_fc']
area = config['area']
sr = config['spatial_reference']

hex_output = os.path.join(hex_orig_folder, hex_gdb, hex_fc)

# orignal hex shp by 4km grids
hex_shp = os.path.join(hex_root, 'HEX_ORIG', 'GRID_HP_unzipped')

try:
    arcpy.CreateFileGDB_management(hex_orig_folder, hex_gdb)
except:
    pass

arcpy.env.workspace = os.path.join(hex_orig_folder, hex_gdb)
arcpy.env.overwriteOutput = True
arcpy.env.outputCoordinateSystem = arcpy.SpatialReference(sr)

# merge all grid hex into one file
grids_list = os.listdir(hex_shp)
for i, grid in enumerate(grids_list):
    grid_fc = os.path.join(hex_shp, grid, grid + '.shp')
    arcpy.MakeFeatureLayer_management(grid_fc, 'lyr_tmp')
    if i == 0:
        arcpy.CopyFeatures_management('lyr_tmp', hex_output)
        i += 1
    else:
        arcpy.Append_management('lyr_tmp', hex_output)

################################
# prep production data and creat working gdb
work_gdb = os.path.join(hex_root, 'working.gdb')
if arcpy.Exists(work_gdb):
    pass
else:
    print('creating working gdb')
    arcpy.CreateFileGDB_management(hex_root, 'working.gdb')

arcpy.env.workspace = work_gdb
arcpy.env.overwriteOutput = True
arcpy.env.outputCoordinateSystem = arcpy.SpatialReference(sr)
output = os.path.join(work_gdb, area + '_EXPTGRID')
grid_layer = os.path.join(hex_root, 'AreaB_ExportGrid.shp')
arcpy.CopyFeatures_management(grid_layer, output)

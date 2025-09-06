import arcpy
import os
from shared.trees_to_csv_sp_split_ami import *
import time

config = read_yaml_config()
hex_root = config['root_folder']
hex_output_folder = config['hex_output_folder']
hex_gdb = config['hex_gdb']
hex_fc = config['hex_fc']
hex_output = os.path.join(hex_output_folder, hex_gdb, hex_fc + '_merged')

# create output folder
gdb_r13 = os.path.join(hex_output_folder, 'AreaB_R13_HEX_Inventory.gdb')
if arcpy.Exists(gdb_r13):
    pass
else:
    arcpy.CreateFileGDB_management(hex_output_folder, 'AreaB_R13_HEX_Inventory.gdb')

gdb_e14 = os.path.join(hex_output_folder, 'AreaB_E14_HEX_Inventory.gdb')
if arcpy.Exists(gdb_e14):
    pass
else:
    arcpy.CreateFileGDB_management(hex_output_folder, 'AreaB_E14_HEX_Inventory.gdb')

hex_r13 = os.path.join(hex_output_folder, 'AreaB_R13_HEX_Inventory.gdb', 'AMI_AREA_B_R13_Hexagon_Inventory')
hex_e14 = os.path.join(hex_output_folder, 'AreaB_E14_HEX_Inventory.gdb', 'AMI_AREA_B_E14_Hexagon_Inventory')

# 
Start = time.time()
arcpy.env.workspace = os.path.join(hex_output_folder, hex_gdb)
arcpy.env.overwriteOutput = True
arcpy.SplitByAttributes_analysis(hex_output, os.path.join(hex_output_folder, hex_gdb), 'FMU')

# copy the feature class over
arcpy.env.workspace = os.path.join(hex_output_folder, 'AreaB_R13_HEX_Inventory.gdb')
arcpy.env.overwriteOutput = True
arcpy.env.outputCoordinateSystem = arcpy.SpatialReference("NAD 1983 UTM Zone 11N")
r13_input = os.path.join(hex_output_folder, hex_gdb, 'R13')
arcpy.CopyFeatures_management(r13_input, hex_r13)
print('finish copy R13')

arcpy.env.workspace = os.path.join(hex_output_folder, 'AreaB_E14_HEX_Inventory.gdb')
arcpy.env.overwriteOutput = True
arcpy.env.outputCoordinateSystem = arcpy.SpatialReference("NAD 1983 UTM Zone 11N")
e14_input = os.path.join(hex_output_folder, hex_gdb, 'E14')
arcpy.CopyFeatures_management(e14_input, hex_e14)
print('finish copy E14')
End = time.time()
print(round((End - Start)/60, 2), ' mins to finish')
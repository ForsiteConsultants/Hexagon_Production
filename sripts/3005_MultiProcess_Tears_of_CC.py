from shared.trees_to_csv_sp_split_ami import *
import arcpy
from arcpy.sa import *
import os
import pandas as pd
from multiprocessing import Pool
import time

####################
#####  CROWN COVER CALCULATION

def calc_cc_to_hex(chm_1m, grid, processPath, hex_grid_folder, hexid):
    fc = os.path.join(hex_grid_folder, 'HEX_' + grid + '.gdb', grid)

    hex_fc_topht_raster(processPath, fc, area)
    tbl_60 = cc_calc(processPath, fc, chm_1m, area, 60)
    df = feature_class_to_pandas_data_frame(tbl_60, [hexid, 'COUNT', 'SUM']).set_index(hexid)
    df['CC'] = (df['SUM']/df['COUNT'])*100
    df['CC'] = df['CC'].astype(int)
    cc_dict_60 = df['CC'].to_dict()

    with arcpy.da.UpdateCursor(fc, [hexid, 'Crown_Closure']) as cursor:
        for row in cursor:
            if row[0] in cc_dict_60:
                row[1] = cc_dict_60[row[0]]
            
            cursor.updateRow(row)


####################################
config = read_yaml_config()
hex_root = config['root_folder']
hex_output_folder = config['hex_output_folder']
hexid = config['hex_id']
csv_folder = config['csv_folder']
sr = config['spatial_reference']
area = config['area']

arcpy.CheckOutExtension("Spatial")

chm_1m = r'S:\1845\2\03_MappingAnalysisData\02_Data\06_Hexagon_Production\04_CHM\HintonCHM.tif'
processPath=os.path.join(hex_root, 'working_cc.gdb')
if arcpy.Exists(processPath):
    pass
else:
    arcpy.CreateFileGDB_management(hex_root, 'working_cc.gdb')
arcpy.env.workspace = processPath
arcpy.env.outputCoordinateSystem = arcpy.SpatialReference(sr)
arcpy.env.overwriteOutput = True

hex_grid_folder = os.path.join(hex_output_folder, 'GRID')

# grids to be processes
df = pd.read_csv(os.path.join(csv_folder, 'MultiProcessing_files_input_AREA_B.csv'))
grid_list = df.GRID.tolist()
grid_list.sort()
# grid_list = grid_list[:4]
# grid = 'AA_2'


hexid = 'HEXID'

Start = time.time()

for i in range(0, len(grid_list)):
    grid = grid_list[i]
    print(f'process {i}th grid {grid}')
    calc_cc_to_hex(chm_1m, grid, processPath, hex_grid_folder, hexid)

End = time.time()
print ('\n----Process finished---- \n\nProcessing time (s): ', End - Start, '\n')

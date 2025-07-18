import arcpy
import os
from trees_to_csv_sp_split_ami import *
import time
import pandas as pd
from multiprocessing import Pool

config = read_yaml_config()
csv_folder = config['csv_folder']
# update NSR and FMU
nsr = r'S:\1845\2\03_MappingAnalysisData\02_Data\01_Base_Data\basedata.gdb\Area_B_NSR'
hex_orig = r'S:\1845\2\03_MappingAnalysisData\02_Data\06_Hexagon_Production\02_Process\hex_orig\AMI_AREA_B_Hexagon.gdb\AMI_AREA_B_Hexagon'
work_gdb = r'S:\1845\2\03_MappingAnalysisData\02_Data\06_Hexagon_Production\working.gdb'

# ######################
# #####################
# # PART 1
# arcpy.env.workspace = work_gdb
# arcpy.env.overwriteOutput = True

# arcpy.Dissolve_management(nsr, 'nsr_fmu', ['NSRCODE', 'FMU'], multi_part=False)
# print('finish dissolve and start intersect')
# arcpy.PairwiseIntersect_analysis(['nsr_fmu', hex_orig], 'hex_nsr_fmu')
# print('finish intersect with NSR')
# df = feature_class_to_pandas_data_frame('hex_nsr_fmu', ['HEXID', 'NSRCODE', 'FMU'])
# df_sum = df[(~df['HEXID'].isin([0, '0']))&(~df['NSRCODE'].isin([0, '0'])) &(~df['FMU'].isin([0, '0']))].groupby('HEXID')[['NSRCODE', 'FMU']].first()
# print(df_sum.head())
# df_sum.to_csv(os.path.join(csv_folder, 'HEX_FMU_NSR.csv'))
##########################
##########################

########################
########### PART 2 
##### MULTIPROCESSING

# admin_info = os.path.join(csv_folder, 'HEX_FMU_NSR.csv')
# df_dict = pd.read_csv(admin_info).set_index('HEXID').to_dict(orient='index')

def add_info_to_hex(hex_grid_folder, grid, df_dict=None):
    fc = os.path.join(hex_grid_folder, 'HEX_' + grid + '.gdb', grid)
    print('processing grid ', grid)
    try:
        with arcpy.da.UpdateCursor(fc, ['HEXID', 'NSRCODE', 'FMU', 'Crown_Closure', 'TOP_HEIGHT', 'AOI_AREA']) as cursor:
            for row in cursor:
                # # update cc = 0 if no trees present
                # if row[4] == 0 or row[4] is None:
                #     row[3] = 0

                # hexid = row[0]
                # if hexid in df_dict:
                #     row[1] = df_dict[hexid]['NSRCODE']
                row[2] = row[5]
                cursor.updateRow(row)
    except Exception as e:
        print(grid, e)

#######
config = read_yaml_config()
hex_root = config['root_folder']
hex_output_folder = config['hex_output_folder']
hex_grid_folder = os.path.join(hex_output_folder, 'GRID')

# grids to be processes
df = pd.read_csv(os.path.join(csv_folder, 'MultiProcessing_files_input_AREA_B.csv'))
grid_list = df.GRID.tolist()
grid_list.sort()
# grid_list = grid_list[:4]
grid = 'AA_2'

Start = time.time()

# add_info_to_hex(hex_grid_folder, grid, df_dict)

args = [(hex_grid_folder, grid) for grid in grid_list]
if __name__ == '__main__':
    with Pool(processes=4) as pool:
        pool.starmap(add_info_to_hex, args)


End = time.time()
print ('\n----Process finished---- \n\nProcessing time (s): ', End - Start, '\n')

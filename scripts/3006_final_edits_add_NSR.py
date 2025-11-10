import arcpy
import os
from shared.trees_to_csv_sp_split_ami import *
import time
import pandas as pd
from multiprocessing import Pool
from shared.logger_utils import get_logger

logger = get_logger('3006_final_edits_add_NSR')

yml_file = r'S:\1845\5\03_MappingAnalysisData\03_Scripts\06_HexProduction\Hexagon_Production\shared\config.yml'
config = read_yaml_config(yml_file)
csv_folder = config['csv_folder']

# update NSR and FMU
nsr = r'S:\1845\5\03_MappingAnalysisData\02_Data\01_Base_Data\basedata.gdb\Area_G_AVI_Combined_UseForStage2'
hex_orig = r'S:\1845\5\03_MappingAnalysisData\02_Data\06_Hexagon_Production\02_Process\hex_orig\AMI_AREA_G_Hexagon.gdb\AMI_AREA_G_Hexagon'
work_gdb = r'S:\1845\5\03_MappingAnalysisData\02_Data\06_Hexagon_Production\working.gdb'

######################
#####################
# PART 1
arcpy.env.workspace = work_gdb
arcpy.env.overwriteOutput = True


arcpy.Dissolve_management(nsr, 'nsr_AreaG', ['NSRCODE'], multi_part=False)
logger.info('Finished dissolve, starting intersect')
arcpy.PairwiseIntersect_analysis(['nsr_AreaG', hex_orig], 'hex_nsr')
logger.info('Finished intersect with NSR')
df = feature_class_to_pandas_data_frame('hex_nsr', ['HEXID', 'NSRCODE'])
df_sum = df[(~df['HEXID'].isin([0, '0']))&(~df['NSRCODE'].isin([0, '0']))].groupby('HEXID')[['NSRCODE']].first()
logger.info(f"NSR summary head:\n{df_sum.head()}")
df_sum.to_csv(os.path.join(csv_folder, 'HEX_NSR.csv'))
##########################
##########################

########################
########### PART 2 
##### MULTIPROCESSING

admin_info = os.path.join(csv_folder, 'HEX_NSR.csv')
df_dict = pd.read_csv(admin_info).set_index('HEXID').to_dict(orient='index')

def add_info_to_hex(hex_grid_folder, grid, df_dict=None):
    fc = os.path.join(hex_grid_folder, 'HEX_' + grid + '.gdb', grid)
    logger.info(f'Processing grid {grid}')
    try:
        with arcpy.da.UpdateCursor(fc, ['HEXID', 'NSRCODE', 'FMU', 'Crown_Closure', 'TOP_HEIGHT', 'AOI_AREA']) as cursor:
            for row in cursor:
                # update cc = 0 if no trees present
                if row[4] == 0 or row[4] is None:
                    row[3] = 0

                hexid = row[0]
                if hexid in df_dict:
                    row[1] = df_dict[hexid]['NSRCODE']
                row[2] = row[5]
                cursor.updateRow(row)
        logger.info(f'Finished grid {grid}')
    except Exception as e:
        logger.error(f'Error processing grid {grid}: {e}', exc_info=True)

#######
config = read_yaml_config()
hex_root = config['root_folder']
hex_output_folder = config['hex_output_folder']
hex_grid_folder = os.path.join(hex_output_folder, 'GRID')

# grids to be processes
df = pd.read_csv(os.path.join(csv_folder, 'MultiProcessing_files_input_AREA_G.csv'))
grid_list = df.GRID.tolist()
grid_list.sort()
# grid_list = grid_list[:4]
# grid = 'A24'

Start = time.time()

# add_info_to_hex(hex_grid_folder, grid, df_dict)

args = [(hex_grid_folder, grid, df_dict) for grid in grid_list]
if __name__ == '__main__':
    with Pool(processes=8) as pool:
        pool.starmap(add_info_to_hex, args)

End = time.time()
logger.info(f'----Process finished---- Processing time (s): {End - Start}')
import arcpy
import pandas as pd
import time
from shared.trees_to_csv_sp_split_ami import *
from multiprocessing import Pool
from shared.logger_utils import get_logger

logger = get_logger('3004_b_MultiProcess_addFields_treeList')

#Adding fields to the hexagon feature classes to prepare for summarized data
Start = time.time()
# define working directory
yml_file = r'S:\1845\5\03_MappingAnalysisData\03_Scripts\05_HEXAGON_PRODUCTION\config.yml'
config = read_yaml_config(yml_file)
hex_root = config['root_folder']
hex_output_folder = config['hex_output_folder']
hex_gdb = config['hex_gdb']
hex_fc = config['hex_fc']
csv_folder = config['csv_folder']
hex_output = os.path.join(hex_output_folder, hex_gdb, hex_fc)


def multi_add_fields_treelist(output_folder, grid):
    """
    hex_shp_folder: directory for original shapefile
    grid: 4km grid name
    output folder: directory for output gdb by 4km grids
    field_list: fields need to be added
    """
    out_fc = os.path.join(output_folder, 'HEX_' + grid + '.gdb', grid)
    try:
        field_names = [f.name for f in arcpy.ListFields(out_fc)]
        if 'CON_HT_5_6' in field_names:
            logger.info(f"{grid} Field created")
        else:
            logger.info(f'Start adding fields {grid}')
            for sp_type in ['CON', 'DEC']:
                arcpy.AddField_management(out_fc, sp_type + '_HT_'+str(5)+"_"+str(6)+"_COUNT", "SHORT")
                arcpy.AddField_management(out_fc, sp_type + '_HT_'+str(5)+"_"+str(6)+"_DBH", 'FLOAT')
                arcpy.AddField_management(out_fc, sp_type + '_HT_'+str(5)+"_"+str(6)+"_SP", 'TEXT', field_length=30)
                for i in range(6, 50, 2):
                    arcpy.AddField_management(out_fc, sp_type + '_HT_'+str(i)+"_"+str(i+2)+"_COUNT", "SHORT")
                    arcpy.AddField_management(out_fc, sp_type + '_HT_'+str(i)+"_"+str(i+2)+"_DBH", 'FLOAT')
                    arcpy.AddField_management(out_fc, sp_type + '_HT_'+str(i)+"_"+str(i+2)+"_SP", 'TEXT', field_length=30)
            logger.info(f"Finished adding fields for {grid}")
    except Exception as e:
        logger.error(f"Error processing {grid}: {e}", exc_info=True)


##############################################
######## define path
area = 'AREA_G'

# get the list of grids to process
multiprocess = os.path.join(csv_folder, 'MultiProcessing_files_input_' + area + '.csv')
grid_list = pd.read_csv(multiprocess).GRID.unique()
grid_list.sort()

Start = time.time()


###################3
##### test function
# grid = 'A16'
# multi_add_fields_treelist(output_folder, grid)

#############################
#### multi process
output_folder = r'S:\1845\5\03_MappingAnalysisData\02_Data\06_Hexagon_Production\02_Process\hex_output\GRID'
args = [(output_folder, grid) for grid in grid_list]
cores = 10

if __name__ == '__main__':
    with Pool(processes=cores) as pool:
        pool.starmap(multi_add_fields_treelist, args)
End = time.time()
logger.info(f"it takes {round((End - Start)/60, 2)} mins to finish")


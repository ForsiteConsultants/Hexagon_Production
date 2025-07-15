import arcpy
import pandas as pd
import time
from trees_to_csv_sp_split_ami import *
from multiprocessing import Pool

# #Adding fields to the hexagon feature classes to prepare for summarized data
# Start = time.time()
# # define working directory
# config = read_yaml_config()
# hex_root = config['root_folder']
# hex_output_folder = config['hex_output_folder']
# hex_gdb = config['hex_gdb']
# hex_fc = config['hex_fc']
# csv_folder = config['csv_folder']
# hex_output = os.path.join(hex_output_folder, hex_gdb, hex_fc)


def multi_add_fields_treelist(output_folder, grid):
    """
    hex_shp_folder: directory for original shapefile
    grid: 4km grid name
    output folder: directory for output gdb by 4km grids
    field_list: fields need to be added
    """
    gdb_path = os.path.join(output_folder, 'HEX_' + grid + '.gdb')
    out_fc = os.path.join(gdb_path, grid)
    
    print('start adding fields ', grid)
    for sp_type in ['CON', 'DEC']:
        arcpy.AddField_management(out_fc, sp_type + '_HT_'+str(5)+"_"+str(6)+"_COUNT", "SHORT")
        arcpy.AddField_management(out_fc, sp_type + '_HT_'+str(5)+"_"+str(6)+"_DBH", 'FLOAT')
        arcpy.AddField_management(out_fc, sp_type + '_HT_'+str(5)+"_"+str(6)+"_SP", 'TEXT', field_length=30)

        for i in range(6, 50, 2):
            arcpy.AddField_management(out_fc, sp_type + '_HT_'+str(i)+"_"+str(i+2)+"_COUNT", "SHORT")
            arcpy.AddField_management(out_fc, sp_type + '_HT_'+str(i)+"_"+str(i+2)+"_DBH", 'FLOAT')
            arcpy.AddField_management(out_fc, sp_type + '_HT_'+str(i)+"_"+str(i+2)+"_SP", 'TEXT', field_length=30)


##############################################
######## define path
output_folder = r'S:\1845\2\03_MappingAnalysisData\02_Data\06_Hexagon_Production\02_Process\hex_output\GRID'
area = 'AREA_B'

# get the list of grids to process
multiprocess = r'S:\1845\2\03_MappingAnalysisData\02_Data\06_Hexagon_Production\02_Process\csv_output\MultiProcessing_files_input_' + area + '.csv'
grid_list = pd.read_csv(multiprocess).GRID.unique()
# grid_list = grid_list[0:2]
grid_list.sort()
# grid = 'AA_2'
args = [(output_folder, grid) for grid in grid_list]

Start = time.time()

if __name__ == '__main__':
    with Pool(processes=8) as pool:
        pool.starmap(multi_add_fields_treelist, args)

End = time.time()

print(round((End - Start)/60, 2), ' mins to finish')


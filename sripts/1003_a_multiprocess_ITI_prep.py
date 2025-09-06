import arcpy
from shared.trees_to_csv_sp_split_ami import *

config = read_yaml_config()
hex_root = config['root_folder']
hex_orig_folder = config['hex_orig_folder']
hex_output_folder = config['hex_output_folder']
hex_gdb = config['hex_gdb']
hex_fc = config['hex_fc']
sr = config['spatial_reference']
hex_orig = os.path.join(hex_orig_folder, hex_gdb, hex_fc)
hex_output = os.path.join(hex_output_folder, hex_gdb, hex_fc)
iti_root = config['iti_gdb']
csv_folder = config['csv_folder']

area = 'AREA_B'

# list of tuples that zip arguments for multi processing
df_list = []
#get key paths to data and outputs based on processing group

if arcpy.Exists(hex_orig):
    # get tree gdb
    treesGDB = iti_root
    hex_ar = feature_class_to_pandas_data_frame(hex_orig, ['EXPTGRIDID'])

    # get unique list of grids to process in this processing group
    gridList = hex_ar.EXPTGRIDID.unique()
    gridList.sort()
    print(gridList)
    arcpy.env.workspace = treesGDB
    arcpy.env.overwriteOutput = False

    i = 0  #**** use this to restart a process if interuppted, make sure 0 if starting new
    endAt = len(gridList)
    # endAt = 5
    
    for x in range(i, endAt):
        row = [csv_folder]
        grid = gridList[x]
        fc_iti = os.path.join(treesGDB, os.path.basename(treesGDB) + '.table_' + gridList[x].lower())
        row.append(grid)
        row.append(fc_iti)
        # single item tuple needs to followed by a comma
        df_list.append(row)

    df = pd.DataFrame(df_list, columns = ['CSV_FOLDER', 'GRID', 'ITI_PATH'])
    df.to_csv(os.path.join(csv_folder, 'MultiProcessing_files_input_' + area + '.csv'), index=False)
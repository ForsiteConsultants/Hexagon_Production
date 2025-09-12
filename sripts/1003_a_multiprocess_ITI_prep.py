import arcpy
from shared.trees_to_csv_sp_split_ami import *
from shared.logger_utils import get_logger

logger = get_logger('1003_a_multiprocess_ITI_prep')

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

area = 'AREA_G'
hex_grid = r'S:\1845\5\03_MappingAnalysisData\02_Data\06_Hexagon_Production\01_HEX_GRID\unzipped'

# get tree gdb and hex grid gdb
# final hex grid list should come from ITI not hex grids
# because some grids may not have ITI
treesGDB = iti_root

try:
    gridList = [f[5:] for f in os.listdir(hex_grid)]
    gridList.sort()
    logger.info(f"Loaded and sorted gridList: {gridList}")
except Exception as e:
    logger.error(f"Failed to list or sort grids in {hex_grid}: {e}", exc_info=True)
    print(f"Failed to list or sort grids in {hex_grid}: {e}")
    gridList = []

i = 0  #**** use this to restart a process if interuppted, make sure 0 if starting new
endAt = len(gridList)

# empty list to store output
df_list = []

for x in range(i, endAt):
    try:
        row = [csv_folder]
        grid = gridList[x]
        fc_iti = os.path.join(treesGDB, os.path.basename(treesGDB) + '.table_' + gridList[x].lower())
        if arcpy.Exists(fc_iti):
            row.append(grid)
            row.append(fc_iti)
            df_list.append(row)
            logger.info(f"Added ITI for grid {grid}: {fc_iti}")
        else:
            logger.warning(f"ITI does not exist for grid {grid}")
            print(f"ITI does not exist for grid {grid}")
    except Exception as e:
        logger.error(f"Failed processing grid {gridList[x]}: {e}", exc_info=True)
        print(f"Failed processing grid {gridList[x]}: {e}")

try:
    df = pd.DataFrame(df_list, columns = ['CSV_FOLDER', 'GRID', 'ITI_PATH'])
    output_csv = os.path.join(csv_folder, f"MultiProcessing_files_input_{area}.csv")
    df.to_csv(output_csv, index=False)
    logger.info(f"Saved DataFrame to CSV: {output_csv}")
except Exception as e:
    logger.error(f"Failed to save DataFrame to CSV: {e}", exc_info=True)
    print(f"Failed to save DataFrame to CSV: {e}")
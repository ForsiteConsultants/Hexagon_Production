import arcpy
import os
from shared.trees_to_csv_sp_split_ami import *
import time
import pandas as pd
from multiprocessing import Pool

config = read_yaml_config()
csv_folder = config['csv_folder']

# update NSR and FMU
nsr_fc = r'S:\1845\5\03_MappingAnalysisData\02_Data\01_Base_Data\basedata.gdb\Area_G_AVI_Combined_UseForStage2'
hex_orig = r'S:\1845\5\03_MappingAnalysisData\02_Data\06_Hexagon_Production\02_Process\hex_orig\AMI_AREA_G_Hexagon.gdb\AMI_AREA_G_Hexagon'
work_gdb = r'S:\1845\5\03_MappingAnalysisData\02_Data\06_Hexagon_Production\working.gdb'

fmu_fc = r'S:\1845\Client_Admin\Project_Wide_Data\AOI_ABCDEFGH.gdb\AOI_G'

######################
#####################
# # PART 1
# arcpy.env.workspace = work_gdb
# arcpy.env.overwriteOutput = True


# # cleanup old intermediates
# for name in ('nsr_AreaG','fmu_AreaG','hex_nsr','hex_fmu'):
#     if arcpy.Exists(name):
#         arcpy.Delete_management(name)

# # dissolve inputs
# arcpy.Dissolve_management(fmu_fc, 'fmu_AreaG', ['FMU'], multi_part=False)

# # intersect each with the hexagon
# arcpy.PairwiseIntersect_analysis(['fmu_AreaG', hex_orig], 'hex_fmu')

# # helper to summarize one field
# def summarize(field_name):
#     tbl      = f'hex_{field_name.lower()}'
#     df       = feature_class_to_pandas_data_frame(tbl, ['HEXID', field_name])
#     df_clean = df[(df.HEXID!=0) & (df[field_name]!=0)]
#     summary  = df_clean.groupby('HEXID')[[field_name]].first()
#     summary.to_csv(os.path.join(csv_folder, f'HEX_{field_name}.csv'))
#     print(f'wrote HEX_{field_name}.csv')

# summarize('FMU')

# ##########################
# ##########################

########################
########### PART 2 
##### MULTIPROCESSING

admin_info = os.path.join(csv_folder, 'HEX_FMU.csv')
df_dict = pd.read_csv(admin_info).set_index('HEXID').to_dict(orient='index')

def add_info_to_hex(hex_grid_folder, grid, df_dict):
    fc = os.path.join(hex_grid_folder, 'HEX_' + grid + '.gdb', grid)
    print('processing grid', grid)
    fields = ['HEXID', 'FMU', 'Crown_Closure', 'TOP_HEIGHT', 'AOI_AREA']
    try:
        with arcpy.da.UpdateCursor(fc, fields) as cursor:
            for row in cursor:
                hexid      = row[0]
                current_fmu = row[1]
                cc          = row[2]
                height      = row[3]
                area        = row[4]

                # 1) If no area (no trees), set Crown_Closure to 0
                if area in (0, None):
                    row[2] = 0

                # 2) If we have a new FMU for this hexid, overwrite it
                if hexid in df_dict:
                    row[1] = df_dict[hexid]['FMU']

                # write the changes
                cursor.updateRow(row)

    except Exception as e:
        print(f"Error processing {grid}: {e}")


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
args = [
    (hex_grid_folder, grid, df_dict)
    for grid in grid_list
]

if __name__ == '__main__':
    with Pool(processes=8) as pool:
        pool.starmap(add_info_to_hex, args)

        
End = time.time()
print ('\n----Process finished---- \n\nProcessing time (s): ', End - Start, '\n')
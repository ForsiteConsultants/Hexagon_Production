#
#
#  This is the key strip that takes the ITI data organized in grids and rolls-up
#  individual tree info into hexagon level sums, averages, and distribution info
#  for specific attributes. Hexagon attributes are dumped to temporary csv files
#  that are later written to the hexagon feature class. 
#   
#   Dataframes in the script are passed to the trees_to_csv.py script for processing
#   key references: 
#       grouped up plotchar files for additional attributes used in adjustments
#       ITI trees gdb - organized into processing groups and grids
#       hexagon feature class  
#  
from trees_to_csv_sp_split_ami import *
import os
import time
import warnings
from multiprocessing import Pool

warnings.filterwarnings("ignore")

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
hexid = 'HEXID'

iti_fields = [hexid, 'HEIGHT', 'GROSS_VOL', 'GROSS_MVOL', 'NETMERCHVO', 'BASAL_AREA', 'SPECIES', 
              'BIOMASS', 'DBH', 'LOC_DENSTY', 'AVG_TR_HGT', 'CANOPYAREA', 'DWB_FACTOR', 'FMU', 'NSRCODE']
iti_fields = [i.lower() for i in iti_fields]


con_sp = ['sw', 'pl', 'sb', 'fb', 'lt']
dec_sp = ['aw', 'bw', 'pb']

# get the list of grids to process
multiprocess = r'S:\1845\2\03_MappingAnalysisData\02_Data\06_Hexagon_Production\02_Process\csv_output\MultiProcessing_files_input_' + area + '.csv'
grid_list = pd.read_csv(multiprocess).GRID.unique()
# grid_list = grid_list[0:1]
grid_list.sort()
print(grid_list)
# grid_list = ['B1']

def iti_to_csv(csv_folder, grid, iti_gdb, iti_fields, con_sp, dec_sp, hexid = 'HEXID'):
    '''
    This function is the same as 1003_ITI_to_csv script,
    which is used for multiprocessing
    it reads in 4km ITI trees and output will be csv files
    
    Input:
    csv_folder: path to store csv output
    grid: name of 4km grids (string)
    iti_fields: fields from ITI

    Output:
    1. summary csv
    2. ITI tree list for assign species to predicted tree list
    '''
    
    ot_direct = os.path.join(csv_folder, grid)
    os.makedirs(ot_direct, exist_ok=True)
    # print(grid)
    print("Processing Grid:{} for group".format(grid))
    fc_iti = os.path.join(iti_gdb, os.path.basename(iti_gdb) + '.table_' + grid.lower())

    try:
        field_names = [field.name for field in arcpy.ListFields(fc_iti)]
        error_fields = [i for i in iti_fields if i not in field_names]
            
        t1 = feature_class_to_pandas_data_frame(fc_iti, iti_fields)
        # select all trees (ignore height cutoff)
        t1.columns = [i.upper() for i in list(t1.columns.values)]
        t1['HEX_AREA_M2'] = 400

        t1 = t1[(t1['HEIGHT'] < 60) & (t1[hexid] != '0')]

        # get the ITI tree list for tree list process
        treelist = t1[[hexid, 'SPECIES', 'HEIGHT', 'DBH']]
        treelist['LIVE_DEAD'] = np.where(treelist['SPECIES'].isin(['dp', 'sn']), 'D', 'L') 
        treelist = treelist.rename(columns={"SPECIES": "SPP"})
        treelist.to_csv(os.path.join(ot_direct, grid + "_ITI_treelist.csv"), index=False)

        # get the admin fields from ITI
        admin = t1[[hexid, 'FMU', 'NSRCODE']]
        admin_sum = admin.groupby([hexid])[['FMU', 'NSRCODE']].first()
        admin_sum.to_csv(os.path.join(ot_direct, grid + '_admin_fields.csv'), index=True)

        ###########
        # Calculate total stuff
        if len(t1[hexid]) > 0:
            df_multi = addInfoToTrees(t1)
            sum_df = hexRollupSums(df_multi)
            avg_df = hexRollupAverages(df_multi)
            ht_df = hexHeightInfo(df_multi)
            spp_df = hexSpeciesCrosstabs(df_multi)

            # merge all to one df and write all of the above to 1 file
            ot_df_total = sum_df.merge(
                ht_df, how='outer', on=hexid).merge(
                    spp_df, how='outer', on=hexid).merge(
                        avg_df, how='outer', on=hexid)

            ot_df_total.to_csv(os.path.join(ot_direct, "OUTPUT_SUM_TOTAL_" + grid +".csv"), index=False)


        # CALCULATE METRICS BY CONIFER AND DECIDUOUS
        for sp_grp in ['CON', 'DEC']:
            if sp_grp == 'CON':
                t1_sp = t1[t1['SPECIES'].isin(con_sp)]
            else:
                t1_sp = t1[t1['SPECIES'].isin(dec_sp)]

            if len(t1_sp[hexid]) > 0:
                df_multi = addInfoToTrees(t1_sp)
                sum_df = hexRollupSumsSpecies(df_multi)
                avg_df = hexRollupAveragesSpecies(df_multi)
                ht_df = hexHeightInfo(df_multi)

                # merge all to one df and write all of the above to 1 file
                ot_df = sum_df.merge(ht_df, how='outer', on=hexid).merge(
                    avg_df, how='outer', on=hexid
                ).fillna(0)

                ot_df.to_csv(os.path.join(ot_direct, "OUTPUT_SUM_" + sp_grp + '_' + grid +".csv"), index=False)

    except Exception as e:
        print(f"have error processing {grid}, error code: {e}")  


Start = time.time()

args = [(csv_folder, grid, iti_root, iti_fields, con_sp, dec_sp) for grid in grid_list]

if __name__ == '__main__':
    with Pool(processes=4) as pool:
        pool.starmap(iti_to_csv, args)

    
End = time.time()

print(round((End - Start)/60, 2), ' mins to finish')
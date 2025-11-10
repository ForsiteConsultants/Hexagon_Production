import arcpy
import pandas as pd
import time
from shared.trees_to_csv_sp_split_ami import *
from multiprocessing import Pool
from shared.logger_utils import get_logger

logger = get_logger('3004_c_MultiProcess_treelist_to_hex')


def treelist_to_hex(hex_grid_folder, grid, csv_folder, hexid):
    try:
        plot_multi = 25
        dbh_list = ["5_6"]
        print('start process ', grid)
        for ht_b in range(8, 52, 2):
        # for t in range(8, 42, 2): # should be (8, 52)
            dbh_list.append(str(max(ht_b-2, 5))+"_"+str(ht_b))

        hex_fc = os.path.join(hex_grid_folder, 'HEX_' + grid + '.gdb', grid)
        flist = arcpy.ListFields(hex_fc)
        fdic = {}
        fl = []
        for f in flist:
            fdic[f.name] = flist.index(f)
            fl.append(f.name)

        for t in ['CON', 'DEC']:
            try:
                aws_folder = os.path.join(csv_folder, 'aws_output')
                tl_input = pd.read_csv(os.path.join(aws_folder, grid + '_' + t.lower() + '_treelist_summary.csv'))
                tl_sum_sph = tl_input.groupby(hexid)[['count']].sum()
                tl_sum_sph.columns = ['SPH_LIVE']
                tl_sum_sph *= 25
                # print(tl_sum_sph.head())
                tl_sum_sph_dict = tl_sum_sph.to_dict(orient='index')

                tl_dict = tl_input.set_index([hexid, 'ht_bin']).to_dict(orient='index')
            except:
                # if No species specific trees in that grid
                print(grid, ' does not have ', t, ' ITI')
                raise
                #return
            
            with arcpy.da.UpdateCursor(hex_fc, fl) as cursor:
                for row in cursor:
                    hex_id = row[fdic[hexid]]

                    if hex_id in tl_sum_sph_dict: # if there is live trees in the hex:
                        sph_live = tl_sum_sph_dict[hex_id]['SPH_LIVE']
                        # adjust sph and merch sph
                        if sph_live > row[fdic[t + '_MERCH_SPH']] or int(row[fdic[t + '_MERCH_SPH']] or 0) <= 0:
                            pass
                        else:
                            merch_ratio = row[fdic[t + '_MERCH_SPH']]/row[fdic[t + '_SPH_GT_5m']]
                            sph_merch = sph_live*merch_ratio
                            row[fdic[t + '_MERCH_STEM']] = round(sph_merch/plot_multi)
                            row[fdic[t + '_MERCH_SPH']] = round(sph_merch)
                            if row[fdic[t + '_GMVOL_PRED_HA']] > 0:
                                row[fdic[t + '_NET_VOL_TREE']] = round(row[fdic[t + '_GMVOL_PRED_HA']]/row[fdic[t + '_MERCH_SPH']], 3)
                                row[fdic[t + '_STEM_PER_M3']] = round(1/row[fdic[t + '_NET_VOL_TREE']], 3)
                        row[fdic[t + '_STEM_GT_5m']] = round(sph_live/plot_multi)
                        row[fdic[t + '_SPH_GT_5m']] = round(sph_live)
                        row[fdic[t + '_QMD']] = round(math.sqrt(row[fdic[t + '_BA_HA']]/(0.0000785*sph_live)), 3)

                        
                        # set these to 0
                        row[fdic[t + '_AV_DIAM']] = 0
                        row[fdic[t + '_LOREY_HT']] = 0

                        # assign treelist to hexagon
                        for dd in dbh_list:
                            tl_key = (hex_id, 'HT_' + dd)
                            if tl_key in tl_dict:
                                row[fdic[t + '_HT_' + dd + "_COUNT"]] = tl_dict[tl_key]['count']
                                row[fdic[t + '_HT_' + dd + "_DBH"]] = tl_dict[tl_key]['avg_dbh']
                                row[fdic[t + '_HT_' + dd + "_SP"]] = tl_dict[tl_key]['sp_proportion']
                            else:
                                row[fdic[t + '_HT_' + dd + "_COUNT"]] = 0
                                row[fdic[t + '_HT_' + dd + "_DBH"]] = 0
                                row[fdic[t + '_HT_' + dd + "_SP"]] = None

                        # calculate avg DBH and lorey's height - adjusted by new tree list
                        sum_ba_l = 0
                        for dd in dbh_list:
                            d_fl = int(dd.split("_")[0])
                            d_ci = int(dd.split("_")[1])
                            if int(row[fdic[t + '_HT_'+dd+"_COUNT"]] or 0) > 0:
                                sum_ba_l += math.pi*math.pow((row[fdic[t + '_HT_'+dd+"_DBH"]]/200), 2) * row[fdic[t + '_HT_'+dd+"_COUNT"]]
                        sum_ba_l *= plot_multi

                        if sum_ba_l > 0:
                            rat = row[fdic[t + '_BA_HA']]/sum_ba_l

                            avg_dbh = {'Count': 0, 'dbh': 0}
                            loreys = {'ht_times_ba': 0, 'ba': 0}

                            for dd in dbh_list:
                                d_fl = int(dd.split("_")[0])
                                d_ci = int(dd.split("_")[1])
                                d_md = (d_fl+d_ci)/2.0
                                if int(row[fdic[t + '_HT_'+dd+"_COUNT"]] or 0) > 0:
                                    # adjuste DBH based on BA ratio
                                    dbh_orig = row[fdic[t + '_HT_'+dd+"_DBH"]]
                                    dbh_adj = dbh_orig*math.pow(rat, 0.5)
                                    row[fdic[t + '_HT_'+dd+"_DBH"]] = round(dbh_adj, 3)

                                    # calculate loreys height
                                    ba_l = math.pi*math.pow((row[fdic[t + '_HT_'+dd+"_DBH"]]/200), 2)*int(row[fdic[t + '_HT_'+dd+"_COUNT"]])
                                    loreys['ht_times_ba'] += (d_md) * ba_l
                                    loreys['ba'] += ba_l
                                    
                                    # calculate average dbh
                                    avg_dbh["Count"] += row[fdic[t + '_HT_'+dd+"_COUNT"]]
                                    avg_dbh["dbh"] += row[fdic[t + '_HT_'+dd+"_DBH"]] * row[fdic[t + '_HT_'+dd+"_COUNT"]]


                            row[fdic[t + '_AV_DIAM']] = round(avg_dbh['dbh']/avg_dbh['Count'], 3)
                            if loreys['ba'] > 0:
                                row[fdic[t + '_LOREY_HT']] = round(loreys['ht_times_ba']/loreys['ba'], 3)
                            else:
                                row[fdic[t + '_LOREY_HT']] = 0
                    try:
                        cursor.updateRow(row)
                    except Exception as e:
                        print(grid, hex_id, e)

        ######## PART 2 #################
        dead_df = pd.read_csv(os.path.join(csv_folder, grid, grid + '_DEAD_OUTPUT.csv')).set_index(hexid)
        dead_dict = dead_df.to_dict(orient='index')
        with arcpy.da.UpdateCursor(hex_fc, fl) as cursor:
            for row in cursor:
                hex_id = row[fdic[hexid]]
                row[fdic['LOREY_HT']] = 0
                # potential changes on sph and merch sph
                if hex_id in dead_dict:
                    dead_sph = dead_dict[hex_id]['DEAD_SPH']
                else:
                    dead_sph = 0

                row[fdic['TOTAL_SPH_GT_5m']] = row[fdic['CON_SPH_GT_5m']] + row[fdic['DEC_SPH_GT_5m']] + round(dead_sph)
                row[fdic['TOTAL_STEM_GT_5m']] = row[fdic['CON_STEM_GT_5m']] + row[fdic['DEC_STEM_GT_5m']] + math.ceil(dead_sph/plot_multi)

                row[fdic['TOTAL_MERCH_STEM']] = row[fdic['CON_MERCH_STEM']] + row[fdic['DEC_MERCH_STEM']]
                row[fdic['TOTAL_MERCH_SPH']] = row[fdic['CON_MERCH_SPH']] + row[fdic['DEC_MERCH_SPH']]
                
                if row[fdic['CON_LOREY_HT']] is None:
                    row[fdic['CON_LOREY_HT']] = 0
                    row[fdic['CON_AV_DIAM']] = 0
                if row[fdic['DEC_LOREY_HT']] is None:
                    row[fdic['DEC_LOREY_HT']] = 0
                    row[fdic['DEC_AV_DIAM']] = 0
                
                if int(row[fdic['TOTAL_SPH_GT_5m']] or 0) > 0:
                    loreys = {'ht_times_ba': 0, 'ba': 0}
                    for t in ['CON', 'DEC']:
                        for dd in dbh_list:
                            d_fl = int(dd.split("_")[0])
                            d_ci = int(dd.split("_")[1])
                            d_md = (d_fl+d_ci)/2.0

                            if int(row[fdic[t + '_HT_'+dd+"_COUNT"]] or 0) > 0:
                                ba = math.pi*math.pow((row[fdic[t + '_HT_'+dd+"_DBH"]]/200), 2) * int(row[fdic[t + '_HT_'+dd+"_COUNT"]] or 0)
                                cnt = int(row[fdic[t + '_HT_'+dd+"_COUNT"]] or 0)
                                ht_ba = (d_md) * ba
                                loreys['ht_times_ba'] += ht_ba
                                loreys['ba'] += ba
                    if loreys['ba'] > 0:
                        row[fdic['LOREY_HT']] = round(loreys['ht_times_ba']/loreys['ba'], 3)
                    else:
                        row[fdic['LOREY_HT']] = 0
                    
                    con_lorey = round(row[fdic['CON_LOREY_HT']], 3)
                    dec_lorey = round(row[fdic['DEC_LOREY_HT']], 3)
                    
                    if con_lorey > 0 and dec_lorey == 0:
                        row[fdic['LOREY_HT']] = con_lorey
                    elif con_lorey == 0 and dec_lorey > 0:
                        row[fdic['LOREY_HT']] = dec_lorey
                    else:
                        if row[fdic['LOREY_HT']] > (max(con_lorey, dec_lorey) or row[fdic['LOREY_HT']] < min(con_lorey, dec_lorey)):
                            print(hex_id, ' error')

                else:
                    row[fdic['LOREY_HT']] = 0
                try:
                    cursor.updateRow(row)
                except Exception as e:
                    print(grid, hex_id, e)
    except Exception as e:
        print(grid, hex_id, e)

##################################################################
#Adding fields to the hexagon feature classes to prepare for summarized data
Start = time.time()
# define working directory
yml_file = r'S:\1845\5\03_MappingAnalysisData\03_Scripts\06_HexProduction\Hexagon_Production\shared\config.yml'
config = read_yaml_config(yml_file)
hex_root = config['root_folder']
hex_output_folder = config['hex_output_folder']
hex_gdb = config['hex_gdb']
hex_fc = config['hex_fc']
hexid = 'HEXID'
csv_folder = config['csv_folder']
hex_output = os.path.join(hex_output_folder, hex_gdb, hex_fc)
hex_grid_folder = os.path.join(hex_output_folder, 'GRID')

# grids to be processes
df = pd.read_csv(os.path.join(csv_folder, 'MultiProcessing_files_input_AREA_G.csv'))
grid_list = df.GRID.tolist()
grid_list.sort()
# grid = 'A24'

# ##################
# # test function
# grid = 'A16'
# treelist_to_hex(hex_grid_folder, grid, csv_folder, hexid)

#################33
## multi processing
args = [(hex_grid_folder, grid, csv_folder, hexid) for grid in grid_list]
cores = 10
if __name__ == '__main__':
    with Pool(processes=cores) as pool:
        pool.starmap(treelist_to_hex, args)

End = time.time()

print(round((End - Start)/60, 2), ' mins to finish')

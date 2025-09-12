#
#   
#   This script will start to build the hexagon forest layer by populating 
#   hexagon level data aggregated and adjusted in previous scripts to the fields in the largely 
#   blank hexagon feature class.
#
#   need reference to: primary hexagon feature class, output files with hex attributes
#   everything is organized into hexagon processing groups and 5km grids

import arcpy, os, time
from numpy.lib.type_check import mintypecode
import pandas as pd
import numpy as np
import math
from shared.trees_to_csv_sp_split_ami import *
from multiprocessing import Pool
from shared.logger_utils import get_logger

logger = get_logger('3004_a_MultiProcess_CSV_to_hex')


def csv_to_hex(hex_grid_folder, grid, compiled_grids_folder, csv_folder, hexid):
    spMinTrees = 5
    TOP_HT_THRESH = 7  # top ht must be taller than or equal to this to be adjusted
    plot_multi = 25
    suffix = '_cc_percent'
    spp = ['aw', 'bw', 'pb', 'lt', 'sb', 'sw', 'pl', 'fb', 'dp', 'sn']
    decsp = ['aw', 'pb', 'bw']
    consp = ['lt', 'sb', 'sw', 'pl','fb']
    deadsp = ['dp', 'sn']
    att_list = ['CON_GVOL_PRED_HA', 'CON_GMVOL_PRED_HA', 'CON_NMVOL_PRED_HA', 'CON_SPH_GT_5m', 'CON_STEM_GT_5m', 
                'CON_MERCH_STEM', 'CON_MERCH_SPH', 'CON_BA_HA', 'CON_MBA_HA',
                'DEC_GVOL_PRED_HA', 'DEC_GMVOL_PRED_HA', 'DEC_NMVOL_PRED_HA', 'DEC_SPH_GT_5m', 'DEC_STEM_GT_5m', 
                'DEC_MERCH_STEM', 'DEC_MERCH_SPH', 'DEC_BA_HA', 'DEC_MBA_HA']

    hex_fc = os.path.join(hex_grid_folder, 'HEX_' + grid + '.gdb', grid)

    gen_list = [hexid, 'TOTAL_TREES', 'TOP_HEIGHT', 'GROSS_VOL_HA',	'GROSS_MVOL_HA', 'NETMVOL_HA',
            'BA_ha', 'MBA_ha', 'MERCH_TREES', 'TOTAL_SPH', 'MERCH_SPH', 'DWB_FACTOR']


    flist = arcpy.ListFields(hex_fc)
    fdic = {}
    fl = []
    for f in flist:
        fdic[f.name] = flist.index(f)
        fl.append(f.name)
        
    # Read in the adjustment outputs file
    try:
        hex_adj = pd.read_csv(os.path.join(csv_folder, grid, grid + "_Hex_predicted_output_v5.csv"), low_memory=False).set_index(hexid).fillna(0)
        iti_comp = pd.read_csv(os.path.join(compiled_grids_folder, grid, 'ITI_compile_' + grid + '.csv'), usecols=[hexid, 'MTPM_con', 'MTPM_dec']).set_index(hexid)
    except:
        print(grid, ' does not have ITI')
        #raise

    print(">>> Processing " + grid )
    try:
        arcpy.AlterField_management(hex_fc, 'HEX_ID', 'HEXID', 'HEXID')
        print('field altered')
    except:
        pass
    # this secton only updates con/dec, and ignore dead stuff
    try:
        for t in ['CON', 'DEC']:
            try:
                hex_gen = pd.read_csv(os.path.join(compiled_grids_folder, grid, "OUTPUT_SUM_" + t + '_' + grid + ".csv"), usecols = gen_list, low_memory=False).fillna(0)
            except:
                # if No species specific trees in that grid
                print(grid, ' does not have ', t, ' ITI')
                continue

            # combine original ITI summaries and predicted values into one file
            hex_gen = hex_gen.set_index(hexid).join(hex_adj).join(iti_comp).fillna(0)
            hex_gen = hex_gen.to_dict(orient='index')

            with arcpy.da.UpdateCursor(hex_fc, fl) as cursor:
                for row in cursor:
                    hex_id = row[fdic[hexid]]
                    # current_hex_id = row[fdic[hexid]]
                    # if there are any trees in CON or DEC group
                    if hex_id in hex_gen and hex_gen[hex_id]['TOTAL_TREES'] > 0:
                        # Dan added this because we still get partial hexagons with less than 4 tree points, this sets minimum at 4 actual trees in a hex sample area
                        # use adjusted values if meets ALL 3 conditions
                        # 1. Total trees > 9
                        # 2. Con/dec trees > 5
                        # 3. Con/dec top height > 10m
                        
                        if hex_gen[hex_id]['TOTAL_TREES'] > spMinTrees and hex_gen[hex_id]['TOP_HEIGHT'] >= TOP_HT_THRESH and round(hex_gen[hex_id]['PRED_' + t + '_SPH_GT_5m']/plot_multi) > spMinTrees:
                            gv = min(hex_gen[hex_id]['PRED_' + t + '_GVOL_PRED_HA'], 850)
                            tsp = min(hex_gen[hex_id]['PRED_' + t + '_SPH_GT_5m'], 8000)
                            ts = tsp/plot_multi

                            baha = min(hex_gen[hex_id]['PRED_' + t + '_BA_HA'], 85)
                            if hex_gen[hex_id]['MERCH_TREES'] > spMinTrees:
                                gmv = min(hex_gen[hex_id]['PRED_' + t + '_GMVOL_PRED_HA'], 800)
                                nmv = gmv*(1 - hex_gen[hex_id]['DWB_FACTOR'])
                                # msp = min(hex_gen[hex_id]['PRED_' + t + '_MERCH_SPH'], 2000)
                                tpm = hex_gen[hex_id]['PRED_' + t + '_STEM_PER_M3']
                                msp = tpm*gmv
                                ms = msp/plot_multi
                                m_baha = min(hex_gen[hex_id]['PRED_' + t + '_MBA_HA'], 80)
                                
                                # adjust tpm
                                
                                if tpm < 0.5 or tpm > 20 or msp > 2000:
                                    gmv = min(hex_gen[hex_id]['GROSS_MVOL_HA'], 800)
                                    nmv = min(hex_gen[hex_id]['NETMVOL_HA'], 750)
                                    msp = min(hex_gen[hex_id]['MERCH_SPH'], 2000)
                                    ms = hex_gen[hex_id]['MERCH_TREES']
                                    m_baha = min(hex_gen[hex_id]['MBA_ha'], 80)
                                    # tpm = max(hex_gen[hex_id]['MTPM_' + t.lower()], 0.2)
                                    tpm = max(msp/gmv, 0.2)

                            else:
                                gmv = min(hex_gen[hex_id]['GROSS_MVOL_HA'], 800)
                                nmv = min(hex_gen[hex_id]['NETMVOL_HA'], 750)
                                msp = min(hex_gen[hex_id]['MERCH_SPH'], 2000)
                                ms = hex_gen[hex_id]['MERCH_TREES']
                                m_baha = min(hex_gen[hex_id]['MBA_ha'], 80)
                                # tpm = max(hex_gen[hex_id]['MTPM_' + t.lower()], 0.2)
                                if gmv > 0:
                                    tpm = max(msp/gmv, 0.2)
                                else:
                                    tpm = 0
                        else:

                            gv = min(hex_gen[hex_id]['GROSS_VOL_HA'], 850)
                            gmv = min(hex_gen[hex_id]['GROSS_MVOL_HA'], 800)
                            nmv = min(hex_gen[hex_id]['NETMVOL_HA'], 750)
                            tsp = hex_gen[hex_id]['TOTAL_SPH']
                            ts = hex_gen[hex_id]['TOTAL_TREES']
                            msp = hex_gen[hex_id]['MERCH_SPH']
                            ms = hex_gen[hex_id]['MERCH_TREES']
                            baha = min(hex_gen[hex_id]['BA_ha'], 85)
                            m_baha = min(hex_gen[hex_id]['MBA_ha'], 80)
                            # tpm = max(hex_gen[hex_id]['MTPM_' + t.lower()], 0.2)
                            if gmv > 0:
                                tpm = max(msp/gmv, 0.2)
                            else:
                                tpm = 0

                        if hex_gen[hex_id]['GROSS_MVOL_HA'] > 0:
                            # ADJUST GROSS VOLUME
                            vol_ratio = hex_gen[hex_id]['GROSS_VOL_HA']/hex_gen[hex_id]['GROSS_MVOL_HA']
                            if gmv > gv:
                                gv = min(gmv*vol_ratio, 850)
                                # row[fdic['OVR_GMVOL']] = 1
                                if gv-gmv > 150:
                                    gv = min(gmv + 150, 850)

                        if hex_gen[hex_id]['MERCH_SPH'] > 0:
                            sph_ratio = hex_gen[hex_id]['TOTAL_SPH']/hex_gen[hex_id]['MERCH_SPH']
                            if msp > tsp:
                                # row[fdic['OVR_SPH']] = 1
                                tsp = min(msp*sph_ratio, 8000)
                                if tsp-msp > 2000:
                                    tsp = min(msp+2000, 8000)
                                ts = round(tsp/plot_multi)

                        if hex_gen[hex_id]['MBA_ha'] > 0:
                            baha_ratio = hex_gen[hex_id]['BA_ha']/hex_gen[hex_id]['MBA_ha']
                            if m_baha > baha:
                                # row[fdic['OVR_BA']] = 1
                                baha = min(m_baha*baha_ratio, 80)
                                if baha-m_baha > 25:
                                    baha = min(m_baha + 25, 80)

                        row[fdic[t + '_GVOL_PRED_HA']] = round(gv, 3)
                        row[fdic[t + '_GMVOL_PRED_HA']] = round(gmv, 3)
                        row[fdic[t + '_NMVOL_PRED_HA']] = round(nmv, 3)
                        row[fdic[t + '_STEM_GT_5m']] = round(ts, 0)
                        row[fdic[t + '_SPH_GT_5m']] = round(tsp, 0)
                        row[fdic[t + '_BA_HA']] = round(baha, 3)
                        row[fdic[t + '_DWB_FACTOR']] = round(hex_gen[hex_id]['DWB_FACTOR'], 3)
                        row[fdic[t + '_QMD']] = round(math.sqrt(baha/(0.0000785*tsp)), 2)

                        if msp > 0:
                            row[fdic[t + '_NET_VOL_TREE']] = round(1/tpm, 3)
                            row[fdic[t + '_STEM_PER_M3']] = round(tpm, 2)
                            row[fdic[t + '_MBA_HA']] = round(m_baha, 3)
                            row[fdic[t + '_MERCH_STEM']] = round(ms, 0)
                            row[fdic[t + '_MERCH_SPH']] = int(round(msp))
                        else:
                            row[fdic[t + '_QMD']] = 0
                            row[fdic[t + '_NET_VOL_TREE']] = 0
                            row[fdic[t + '_STEM_PER_M3']] = 0
                            row[fdic[t + '_MBA_HA']] = 0
                            row[fdic[t + '_MERCH_STEM']] = 0
                            row[fdic[t + '_MERCH_SPH']] = 0
                    else:
                        row[fdic[t + '_GVOL_PRED_HA']] = 0
                        row[fdic[t + '_GMVOL_PRED_HA']] = 0
                        row[fdic[t + '_NMVOL_PRED_HA']] = 0
                        row[fdic[t + '_STEM_GT_5m']] = 0
                        row[fdic[t + '_SPH_GT_5m']] = 0
                        row[fdic[t + '_BA_HA']] = 0
                        row[fdic[t + '_QMD']] = 0
                        row[fdic[t + '_NET_VOL_TREE']] = 0
                        row[fdic[t + '_MBA_HA']] = 0
                        row[fdic[t + '_MERCH_STEM']] = 0
                        row[fdic[t + '_MERCH_SPH']] = 0
                        row[fdic[t + '_DWB_FACTOR']] = 0
                    
                    cursor.updateRow(row)

        # print(">>> Processing " + grid +" " + ' WHOLE SPECIES')
        try:
            hex_all = pd.read_csv(os.path.join(compiled_grids_folder, grid, "OUTPUT_SUM_TOTAL_" + grid + ".csv")).set_index(hexid)
            iti_compile = pd.read_csv(os.path.join(compiled_grids_folder, grid, "ITI_compile_" + grid + ".csv"), usecols=[hexid, 'BPHD', 'SPHD', 'GVPHD']).set_index(hexid)
            hex_admin = pd.read_csv(os.path.join(compiled_grids_folder, grid, grid + "_admin_fields.csv")).set_index(hexid)
        except:
            print(grid, ' does not have ITI')
            #raise
            
            
        hex_all = hex_all.join(iti_compile).to_dict(orient='index')
        hex_admin = hex_admin.to_dict(orient='index')
        dead_list = []
        with arcpy.da.UpdateCursor(hex_fc, fl) as cursor:
            for row in cursor:
                hex_id = row[fdic[hexid]]
                sph_con = 0
                sph_dec = 0
                ba_con = 0
                ba_dec = 0

                for field in att_list:
                    if row[fdic[field]] is None:
                        row[fdic[field]] = 0
                if hex_id in hex_admin:
                    row[fdic['FMU']] = row[fdic['AOI_AREA']]
                    row[fdic['NSRCODE']] = hex_admin[hex_id]['NSRCODE']

                if hex_id in hex_all and hex_all[hex_id]["TOTAL_TREES"] > 0:            
                    # get species percentage info
                    m_s = ['', 0]
                    pp = 0
                    for sp in spp:
                        if sp+"_pct" in fl:
                            row[fdic[sp+"_pct"]] = 0
                            if sp+suffix in hex_all[hex_id]:
                                p = hex_all[hex_id][sp+suffix]*100
                                if p > 0 and p < 1:
                                    p = 1
                                p = int(round(p, 0))
                                row[fdic[sp+"_pct"]] = p
                                if m_s[1] < p:
                                    m_s[0] = sp
                                    m_s[1] = p
                                pp += p
                        else:
                            print(f"check {grid} does not have {sp}")
                    if pp != 100:
                        try:
                            row[fdic[m_s[0]+'_pct']] += 100 - pp
                        except:
                            print(hex_id, m_s[0]+'_pct', m_s, pp)
                    row[fdic['LEADING_SPP']] = m_s[0]

                    # adjust total metrics to add in dead stuff
                    # if dead pct = 100%, use original iti dead values
                    if (int(row[fdic['dp_pct']] or 0) + int(row[fdic['sn_pct']] or 0)) > 50:
                        dead_ba = hex_all[hex_id]['BPHD']
                        dead_sph = hex_all[hex_id]['SPHD']
                        dead_vol = hex_all[hex_id]['GVPHD']
                    elif int(row[fdic['dp_pct']] or 0) > 0 or int(row[fdic['sn_pct']] or 0) > 0: # has dead but no more than 50%
                        dead_pct = int(row[fdic['dp_pct']] or 0) + int(row[fdic['sn_pct']] or 0)
                        dead_ba = (row[fdic['CON_BA_HA']] + row[fdic['DEC_BA_HA']])/(100-dead_pct)*dead_pct
                        # except:
                        #     print|(hex_id, row[fdic['dp_pct']], row[fdic['sn_pct']], dead_pct)
                        dead_sph = (row[fdic['CON_SPH_GT_5m']] + row[fdic['DEC_SPH_GT_5m']])/(100-dead_pct)*dead_pct
                        dead_vol = (row[fdic['CON_GVOL_PRED_HA']] + row[fdic['DEC_GVOL_PRED_HA']])/(100-dead_pct)*dead_pct
                    else:
                        dead_ba = 0
                        dead_sph = 0
                        dead_vol = 0
                    
                    sph_con = row[fdic['CON_SPH_GT_5m']]
                    sph_dec = row[fdic['DEC_SPH_GT_5m']]
                    ba_con = row[fdic['CON_BA_HA']]
                    ba_dec = row[fdic['DEC_BA_HA']]

                    dead_item = [hex_id, dead_ba, dead_sph, dead_vol, sph_con, sph_dec, ba_con, ba_dec]
                    dead_list.append(dead_item)

                    # Get Metrics for all trees  
                    row[fdic['TOTAL_GVOL_PRED_HA']] = row[fdic['CON_GVOL_PRED_HA']] + row[fdic['DEC_GVOL_PRED_HA']] + round(dead_vol, 3)
                    row[fdic['TOTAL_GMVOL_PRED_HA']] = row[fdic['CON_GMVOL_PRED_HA']] + row[fdic['DEC_GMVOL_PRED_HA']]
                    row[fdic['TOTAL_NMVOL_PRED_HA']] = row[fdic['CON_NMVOL_PRED_HA']] + row[fdic['DEC_NMVOL_PRED_HA']]

                    row[fdic['TOTAL_SPH_GT_5m']] = row[fdic['CON_SPH_GT_5m']] + row[fdic['DEC_SPH_GT_5m']] + round(dead_sph)
                    row[fdic['TOTAL_STEM_GT_5m']] = row[fdic['CON_STEM_GT_5m']] + row[fdic['DEC_STEM_GT_5m']] + math.ceil(dead_sph/plot_multi)

                    row[fdic['TOTAL_MERCH_STEM']] = row[fdic['CON_MERCH_STEM']] + row[fdic['DEC_MERCH_STEM']]
                    row[fdic['TOTAL_MERCH_SPH']] = row[fdic['CON_MERCH_SPH']] + row[fdic['DEC_MERCH_SPH']]

                    row[fdic['TOTAL_BA_HA']] = row[fdic['CON_BA_HA']] + row[fdic['DEC_BA_HA']] + round(dead_ba, 3)
                    row[fdic['TOTAL_MBA_HA']] = row[fdic['CON_MBA_HA']] + row[fdic['DEC_MBA_HA']]
                    
                    # get con/dec percentage
                    consum = 0
                    decsum = 0
                    deadsum = 0
                    for sp in spp:
                        if sp+"_pct" in fl:
                            if sp in consp:
                                consum += row[fdic[sp+"_pct"]]
                            elif sp in decsp:
                                decsum += row[fdic[sp+"_pct"]]
                            else:
                                deadsum += row[fdic[sp+"_pct"]]
                    
                    if deadsum > 98:
                        row[fdic['STAND_TYPE']] = 'DEAD'
                    else:
                        decsum = decsum/(100-deadsum)*100
                        consum = consum/(100-deadsum)*100
                        if decsum >= 80:
                            row[fdic['STAND_TYPE']] = 'D'
                        elif consum >= 80:
                            row[fdic['STAND_TYPE']] = 'C'
                        elif decsum >= 50:
                            row[fdic['STAND_TYPE']] = 'DC'
                        elif consum >= 50:
                            row[fdic['STAND_TYPE']] = 'CD'
                        else:
                            print('nostand', decsum, consum, decsum, hex_id)

                    row[fdic['TOP_HEIGHT']] = round(hex_all[hex_id]['TOP_HEIGHT'], 2)
                    row[fdic['MAX_HT_ITI']] = round(hex_all[hex_id]['MaxHt'], 2)
                    try:
                        if hex_all[hex_id]['TOP_HEIGHT'] > 17.5 and hex_all[hex_id]['T'] in decsp and hex_all[hex_id]['S'] in consp:
                            row[fdic['Con_u_Dec']] = 'Y'  # hex_gen[hex_id][]
                        else:
                            row[fdic['Con_u_Dec']] = 'N'  # hex_gen[hex_id][]
                    except:
                        row[fdic['Con_u_Dec']] = 'N'  # hex_gen[hex_id][]
                else:
                    row[fdic['TOTAL_GVOL_PRED_HA']] = 0
                    row[fdic['TOTAL_GMVOL_PRED_HA']] = 0
                    row[fdic['TOTAL_NMVOL_PRED_HA']] = 0

                    row[fdic['TOTAL_SPH_GT_5m']] = 0
                    row[fdic['TOTAL_STEM_GT_5m']] = 0

                    row[fdic['TOTAL_MERCH_STEM']] = 0
                    row[fdic['TOTAL_MERCH_SPH']] = 0

                    row[fdic['TOTAL_BA_HA']] = 0
                    row[fdic['TOTAL_MBA_HA']] = 0
                    
                    row[fdic['LEADING_SPP']] = ''
                    for sp in spp:
                        if sp+"_pct" in fl:
                            row[fdic[sp+"_pct"]] = 0
                    row[fdic['TOP_HEIGHT']] = 0
                    row[fdic['MAX_HT_ITI']] = 0
                    row[fdic['Con_u_Dec']] = ''
                try:
                    cursor.updateRow(row)
                except Exception as e:
                    print(hex_id, grid, e)
        
        df = pd.DataFrame(dead_list, columns = [hexid, 'DEAD_BA',  'DEAD_SPH', 'DEAD_VOL', 'SPH_con', 'SPH_dec', 'BPH_con',  'BPH_dec'])
        df.to_csv(os.path.join(csv_folder, grid, grid + '_DEAD_OUTPUT_v5.csv'), index=False)
        print(f'{grid} processing complete')
    except Exception as e:

        print(grid, e)
        

# ####################################################
config = read_yaml_config()
hex_root = config['root_folder']
hex_output_folder = config['hex_output_folder']
hexid = 'HEXID'
csv_folder = config['csv_folder']
compiled_grids_folder = config['compiled_grids_folder']

# ######################
# #######################

# grids to be processed:
hex_grid_folder = os.path.join(hex_output_folder, 'GRID')
df = pd.read_csv(os.path.join(csv_folder, 'MultiProcessing_files_input_AREA_G.csv'))
grid_list = df.GRID.tolist()
grid_list.sort()


# ##################
# ### test function
Start = time.time()
grid = 'AB29'
csv_to_hex(hex_grid_folder, grid, compiled_grids_folder, csv_folder, hexid)


# ###################3
# #### run 
# args = [(hex_grid_folder, grid, compiled_grids_folder, csv_folder, hexid) for grid in grid_list]
# cores = 10

# if __name__ == '__main__':
#     with Pool(processes=cores) as pool:
#         pool.starmap(csv_to_hex, args)

End = time.time()

print(round((End - Start)/60, 2), ' mins to finish')




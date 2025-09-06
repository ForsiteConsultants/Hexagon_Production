from shared.trees_to_csv_sp_split_ami import *
from multiprocessing import Pool
import time


#############
#### FOR 1845-2 THIS SCRIPT IS RUN ON AWS BATCH  !!!!
# Function to calculate the proportion of species
def calculate_proportion(series):
    sp_counts = series.value_counts().sort_values(ascending=False)
    total = sp_counts.sum()
    raw_proportions = [(t, c / total * 100) for t, c in sp_counts.items()]
    rounded_proportions = [(t, round(p)) for t, p in raw_proportions]

    # Adjust rounding to ensure proportions add up to 100
    diff = 100 - sum(p for _, p in rounded_proportions)
    if diff != 0 and rounded_proportions:
        rounded_proportions[0] = (rounded_proportions[0][0], rounded_proportions[0][1] + diff)

    proportions = ''.join(f"{t}{p}" for t, p in rounded_proportions)
    return proportions

# Define height bins
bins = []
num = 5
while num <= 50:
    bins.append(num)
    num += 1 if num == 5 else 2
labels = ['HT_' + str(i) + '_' + str(i+2) for i in bins]
labels = labels[:-1]
labels[0] = 'HT_5_6'

def treelist_summary(csv_folder, grid, hexid='HEXID', con=['fb', 'lt', 'pl', 'sb', 'sw'], dec = ['aw', 'bw', 'pb']):
    iti_treelist = pd.read_csv(os.path.join(csv_folder, grid, grid + '_ITI_treelist.csv'))
    for type in ['con', 'dec']:
        if type == 'con':
            iti_tmp = iti_treelist[iti_treelist['SPP'].isin(con)]
        else:
            iti_tmp = iti_treelist[iti_treelist['SPP'].isin(dec)]
        iti_tmp = iti_tmp[[hexid, 'DBH', 'SPP', 'HEIGHT']]
        iti_tmp.columns = [hexid, 'DBH', 'SPECIES', 'HEIGHT']
        iti_tmp.SPECIES = iti_tmp['SPECIES'].str.upper()
        iti_tmp['ht_bin'] = pd.cut(iti_tmp['HEIGHT'], bins=bins, labels=labels, include_lowest=True)
        # Group by ID and weight_bin, then calculate statistics
        iti_sum = (
            iti_tmp.groupby([hexid, 'ht_bin'])
            .agg(
                avg_dbh=('DBH', 'mean'),
                sp_proportion=('SPECIES', calculate_proportion),
                count=('SPECIES', 'size')
            )
            .reset_index()
        )
        iti_sum = iti_sum[iti_sum['count'] > 0]

        # get the list of hexid where stem count <= 5
        sph_lt5 = iti_tmp.groupby(hexid)[['DBH']].count()
        sph_lt5 = sph_lt5[sph_lt5['DBH'] <= 5].reset_index()
        hex_lt5 = sph_lt5[hexid].unique()
        # get the full treelist
        iti_hexlist = iti_tmp[hexid].unique()

        # read in predicted tree list
        treelist = os.path.join(csv_folder, grid, 'treelist_output', grid + '_final_treelist_' + type + '_adj.csv')
        if os.path.exists(treelist):
            # get the list of hexid where predicted tree count > 7500 sph
            # treelist script needs to be fixed later
            # this is a quick fix
            tl_df = pd.read_csv(treelist, usecols=[hexid, 'SPECIES', 'HT', 'DBH'])
            sph_gt8k = tl_df.groupby(hexid)[['SPECIES']].count()
            sph_gt8k = sph_gt8k[sph_gt8k['SPECIES'] > 300].reset_index()
            hex_gt8k = sph_gt8k[hexid].unique()


            tl_df['SPECIES'] = tl_df['SPECIES'].str.upper()
            # print(tl_df.head())

            tl_df['ht_bin'] = pd.cut(tl_df['HT'], bins=bins, labels=labels, include_lowest=True)
            # print(tl_df.head())

            # Group by ID and weight_bin, then calculate statistics
            tl_sum = (
                tl_df.groupby([hexid, 'ht_bin'])
                .agg(
                    avg_dbh=('DBH', 'mean'),
                    sp_proportion=('SPECIES', calculate_proportion),
                    count=('SPECIES', 'size')
                )
                .reset_index()
            )
            tl_sum = tl_sum[tl_sum['count'] > 0]
            tl_hexlist = tl_sum[hexid].unique()

        
            hexlist_exclude = [i for i in tl_hexlist if i not in iti_hexlist] # hexlist to exclude from prediction
            hexlist_exclude.extend(hex_lt5)
            hexlist_exclude.extend(hex_gt8k)

            hexlist_add = [i for i in iti_hexlist if i not in tl_hexlist]

            # get treelist from iti if original stem count less than 5 or not in predicted treelist or pred stems too large
            iti_sum_cut = iti_sum[(iti_sum[hexid].isin(hex_lt5))|(iti_sum[hexid].isin(hexlist_add))|(iti_sum[hexid].isin(hex_gt8k))]
            tl_tmp = tl_sum[~tl_sum[hexid].isin(hexlist_exclude)]
            tl_final = pd.concat([iti_sum_cut, tl_tmp])
        
        else:
            tl_final = iti_sum
        hex_new = tl_final[hexid].unique()
        if len(hex_new) == len(iti_hexlist):
            pass
        else:
            print(grid, 'hexid length not match, check!', len(iti_hexlist), len(hex_new))
        tl_final.to_csv(os.path.join(csv_folder, grid, grid + '_' + type + '_treelist_summary.csv'), index=False)



##############################
config = read_yaml_config()
hex_root = config['root_folder']
hex_output_folder = config['hex_output_folder']
hex_gdb = config['hex_gdb']
hex_fc = config['hex_fc']
hex_output = os.path.join(hex_output_folder, hex_gdb, hex_fc)
hexid = 'HEXID'

csv_folder = config['csv_folder']

multiprocess = os.path.join(csv_folder, 'MultiProcessing_files_input_AREA_B.csv')
grid_list = pd.read_csv(multiprocess).GRID.unique()
grid_list.sort()


# grid_list = grid_list[0:5]
grid = 'AF_20'

Start = time.time()
treelist_summary(csv_folder, grid)

# args = [(csv_folder, grid) for grid in grid_list]
# if __name__ == '__main__':
#     with Pool(processes=4) as pool:
#         pool.starmap(treelist_summary, args)

End = time.time()

print(round((End - Start)/60, 2), ' mins to finish')
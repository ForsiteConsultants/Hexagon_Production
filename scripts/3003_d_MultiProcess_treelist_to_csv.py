import yaml
import os
from treelist_creation_al.tl_prediction import *
from treelist_creation_al.preparing_treelist import *
from treelist_creation_al.adj_sph_treelist import *

############
############
## THIS SCRIPT THE 3RD PART ADJ_SPH_TREELIST IS RUNNING ON AWS
### WILL RUN THE WHOLE THING ON AWS NEXT TIME
### THIS FIRST TWO PART IS OKAY...DOABLE ON PROCESSING COMPUTER
### BUT THE LAST PART TAKES FOREVER TO RUN, AND THE PROCESSING TIME UNPREDICTABLE
from multiprocessing import Pool
import warnings
warnings.filterwarnings("ignore")
import yaml
import multiprocessing
import io
import contextlib

def treelist_to_csv(csv_folder, grid, config, id_name = 'HEXID'):

    output_folder = os.path.join(csv_folder, grid, 'treelist_output')
    os.makedirs(output_folder, exist_ok=True)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            tl_prediction(csv_folder, grid, config)
            con_sp = config['con_sp']
            dec_sp = config['dec_sp']
            preparing_treelist(csv_folder, grid, con_sp, dec_sp, id_name)
            # adj_sph_treelist(csv_folder, grid, id_name)
    except Exception as e:
        print(grid, e)
        return None
    
    

################################
################################
config_file = r'S:\1845\5\03_MappingAnalysisData\03_Scripts\06_HexProduction\Hexagon_Production\shared\config.yml'
with open(config_file, 'r') as file:
	config = yaml.safe_load(file)
multiprocess = r'S:\1845\5\03_MappingAnalysisData\02_Data\06_Hexagon_Production\02_Process\csv_output\MultiProcessing_files_input_AREA_G.csv'
grid_list = pd.read_csv(multiprocess).GRID.unique()
grid_list.sort()

csv_folder = config['csvFolderPath']
id_name = config['id_name']

startTime = time.time()

treelist_to_csv(csv_folder, 'AF_20', config)

# args = [(csv_folder, grid, config) for grid in grid_list]
# if __name__ == '__main__':
#     with Pool(processes=30) as pool:
#         pool.starmap(treelist_to_csv, args)


endTime = time.time()
print ('\n----Process finished---- \n\nProcessing time (s): ', endTime - startTime, '\n')
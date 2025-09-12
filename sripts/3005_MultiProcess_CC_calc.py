import arcpy
from arcpy.sa import *
import os
import pandas as pd
from multiprocessing import Pool, Manager
import time
import yaml
import traceback
from functools import partial
from shared.logger_utils import get_logger

logger = get_logger('3005_MultiProcess_CC_calc')

def hex_fc_topht_raster(path, Hex_FC, suffix):
    """
    This function convert TOP_HEIGHT from hexagons to raster file
    """
    output = os.path.join(path, "TOP_HEIGHT_" + suffix)
    arcpy.conversion.PolygonToRaster(Hex_FC, "TOP_HEIGHT", output, "CELL_CENTER", "NONE", 1)

def cc_calc(path, Hex_FC, CHM, suffix, perc=60, hexid='HEXID'):
    """
    This function calculates crown cover using top height
    """
    tpht = Raster(path + r"\TOP_HEIGHT_"+suffix)
    out60pct = tpht * (perc/100)
    out60pct.save(path + r"\pct"+str(perc)+"_"+suffix)
    outCon = Con(Raster(CHM) >= Raster(path + "/pct"+str(perc)+"_"+suffix), 1, 0)
    outCon2 = Con(Raster(CHM) >= 3, outCon, 0)
    outCon2.save(path + "/gt"+str(perc)+"pct_TopHt_"+suffix)
    ZonalStatisticsAsTable(Hex_FC, hexid, path + "/gt"+str(perc)+"pct_TopHt_"+suffix, 
                          path+r'\CellContribute_CC_'+str(perc)+"_"+suffix, statistics_type="SUM")
    return path+r'\CellContribute_CC_'+str(perc)+"_"+suffix

def read_yaml_config():
    """
    Read yaml config and return dictionary of items
    """
    yml_file = r'S:\1845\5\03_MappingAnalysisData\03_Scripts\06_HexProduction\config_hex_G.yml'
    with open(yml_file, 'r') as file:
        config = yaml.safe_load(file)
        return config
    
def feature_class_to_pandas_data_frame(feature_class, field_list):
    return pd.DataFrame(
        arcpy.da.FeatureClassToNumPyArray(
            in_table=feature_class,
            field_names=field_list,
            skip_nulls=False,
            null_value=0
        )
    )

def process_single_grid(grid, config, progress_dict, failed_grids):
    """
    Process a single grid with error handling and progress tracking
    """
    try:
        hex_root = config['root_folder']
        hex_output_folder = config['hex_output_folder']
        hexid = config['hex_id']
        area = config['area']
        sr = config['spatial_reference']
        chm_1m = r"L:\Projects\1845\Area_G_CHM_HR_OVERVIEW\CHM_HR_OVERVIEW_COG.tif"

        logger.info(f"Processing grid {grid}")
        # Create a unique working GDB for this process
        process_gdb = f"working_cc_{grid}.gdb"
        processPath = os.path.join(hex_root, process_gdb)

        if not arcpy.Exists(processPath):
            arcpy.CreateFileGDB_management(hex_root, process_gdb)
            logger.info(f"Created GDB for grid {grid}")

        arcpy.env.workspace = processPath
        arcpy.env.outputCoordinateSystem = arcpy.SpatialReference(sr)
        arcpy.env.overwriteOutput = True

        hex_grid_folder = os.path.join(hex_output_folder, 'GRID')
        fc = os.path.join(hex_grid_folder, 'HEX_' + grid + '.gdb', grid)

        # Process the grid
        hex_fc_topht_raster(processPath, fc, area)
        logger.info(f"Created TOP_HEIGHT raster for grid {grid}")
        tbl_60 = cc_calc(processPath, fc, chm_1m, area, 60)
        logger.info(f"Calculated crown cover for grid {grid}")

        df = feature_class_to_pandas_data_frame(tbl_60, [hexid, 'COUNT', 'SUM']).set_index(hexid)
        df['CC'] = (df['SUM']/df['COUNT'])*100
        df['CC'] = df['CC'].astype(int)
        cc_dict_60 = df['CC'].to_dict()

        with arcpy.da.UpdateCursor(fc, [hexid, 'Crown_Closure']) as cursor:
            for row in cursor:
                if row[0] in cc_dict_60:
                    row[1] = cc_dict_60[row[0]]
                cursor.updateRow(row)
        logger.info(f"Updated Crown_Closure for grid {grid}")

        # Clean up temporary GDB
        arcpy.Delete_management(processPath)
        logger.info(f"Deleted temp GDB for grid {grid}")

        progress_dict[grid] = "Completed"
        return True

    except Exception as e:
        progress_dict[grid] = f"Failed: {str(e)}"
        failed_grids.append(grid)
        logger.error(f"Error processing grid {grid}: {traceback.format_exc()}", exc_info=True)
        print(f"Error processing grid {grid}: {traceback.format_exc()}")
        return False

def init_worker(shared_progress, shared_failed):
    global progress_dict, failed_grids
    progress_dict = shared_progress
    failed_grids = shared_failed

def main():
    if arcpy.CheckExtension("Spatial") == "Available":
        arcpy.CheckOutExtension("Spatial")
    else:
        logger.error("Spatial Analyst extension is not available")
        raise RuntimeError("Spatial Analyst extension is not available")

    config = read_yaml_config()
    csv_folder = config['csv_folder']

    # Read grid list
    df = pd.read_csv(os.path.join(csv_folder, 'MultiProcessing_files_input_AREA_G.csv'))
    grid_list = df.GRID.tolist()
    grid_list = sorted([str(x) for x in grid_list if str(x) != '0'])

    # Set up parallel processing
    num_cores = min(8, os.cpu_count())  # Safe limit as suggested
    manager = Manager()
    progress_dict = manager.dict()
    failed_grids = manager.list()

    # Initialize progress dictionary
    for grid in grid_list:
        progress_dict[grid] = "Pending"

    # Create partial function with fixed config
    process_func = partial(process_single_grid,
                          config=config,
                          progress_dict=progress_dict,
                          failed_grids=failed_grids)

    start_time = time.time()

    logger.info(f"Starting processing of {len(grid_list)} grids with {num_cores} cores...")
    print(f"Starting processing of {len(grid_list)} grids with {num_cores} cores...")

    with Pool(processes=num_cores, initializer=init_worker,
             initargs=(progress_dict, failed_grids)) as pool:
        results = pool.map(process_func, grid_list)

    # Print summary
    logger.info("\nProcessing Summary:")
    print("\nProcessing Summary:")
    for grid, status in progress_dict.items():
        logger.info(f"{grid}: {status}")
        print(f"{grid}: {status}")

    if failed_grids:
        logger.warning("\nFailed grids:")
        print("\nFailed grids:")
        for grid in failed_grids:
            logger.warning(grid)
            print(grid)
    else:
        logger.info("\nAll grids processed successfully.")
        print("\nAll grids processed successfully.")

    total_time = time.time() - start_time
    logger.info(f"\nTotal processing time: {total_time:.2f} seconds")
    logger.info(f"Average time per grid: {total_time/len(grid_list):.2f} seconds")
    print(f"\nTotal processing time: {total_time:.2f} seconds")
    print(f"Average time per grid: {total_time/len(grid_list):.2f} seconds")

if __name__ == '__main__':
    main()

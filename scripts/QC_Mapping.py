import geopandas as gpd
import matplotlib.pyplot as plt
import os
import time
from multiprocessing import Pool
from shared.logger_utils import get_logger

logger = get_logger('3010_GeoPandas_QC')

# ---------------------------------------------------
# Load hex layer once (outside multiprocessing)
# ---------------------------------------------------
gdb_path = r"S:\1845\6\03_MappingAnalysisData\02_Data\06_Hexagon_Production\02_Process\hex_orig\AMI_AREA_H_Hexagon.gdb"
layer_name = "AMI_AREA_H_hexagon_merged"
output_root = r"S:\1845\6\03_MappingAnalysisData\04_Plotfiles\QC_HEX"

logger.info("Loading GeoDataFrame...")
gdf = gpd.read_file(gdb_path, layer=layer_name)
logger.info(f"Loaded {len(gdf)} hexagons with {len(gdf.columns)} fields.")


# ---------------------------------------------------
# Function to create one QC map
# ---------------------------------------------------
def create_qc_map(field):
    try:
        logger.info(f"Processing field: {field}")

        # Skip fields not present
        if field not in gdf.columns:
            logger.warning(f"Field {field} not found in dataset — skipping.")
            return

        # Output folder
        area_folder = os.path.join(output_root, "AMI_AREA_H", "jpeg")
        os.makedirs(area_folder, exist_ok=True)

        # Plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        gdf.plot(column=field,
                 cmap="viridis",   # default colormap
                 linewidth=0,
                 legend=True,
                 ax=ax)

        ax.set_title(f"{field} — AMI_AREA_H", fontsize=14)
        ax.set_axis_off()

        # Save JPEG
        out_file = os.path.join(area_folder, f"{field}.jpg")
        plt.savefig(out_file, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"Exported JPEG for field: {field}")

    except Exception as e:
        logger.error(f"Error plotting {field}: {e}", exc_info=True)


# ---------------------------------------------------
# Fields to process
# ---------------------------------------------------
field_list = ['Crown_Closure', 'CON_AV_DIAM', 'DEC_AV_DIAM', 'CON_LOREY_HT', 'DEC_LOREY_HT',
              'CON_DWB_FACTOR', 'DEC_DWB_FACTOR', 'TOP_HEIGHT', 'MAX_HT_ITI',
              'DEC_NET_VOL_TREE', 'CON_STEM_PER_M3', 'DEC_STEM_PER_M3', 'CON_NET_VOL_TREE',
              'TOTAL_SPH_GT_5m', 'CON_SPH_GT_5m', 'DEC_SPH_GT_5m',
              'TOTAL_MERCH_SPH', 'CON_MERCH_SPH', 'DEC_MERCH_SPH',
              'TOTAL_GVOL_PRED_HA', 'CON_GVOL_PRED_HA', 'DEC_GVOL_PRED_HA',
              'TOTAL_GMVOL_PRED_HA', 'CON_GMVOL_PRED_HA', 'DEC_GMVOL_PRED_HA',
              'TOTAL_NMVOL_PRED_HA', 'CON_NMVOL_PRED_HA', 'DEC_NMVOL_PRED_HA',
              'TOTAL_BA_HA', 'CON_BA_HA', 'DEC_BA_HA',
              'TOTAL_MBA_HA', 'CON_MBA_HA', 'DEC_MBA_HA',
              'CON_QMD', 'DEC_QMD',
              'aw_pct','bw_pct','pb_pct','fb_pct','pl_pct','lt_pct',
              'sw_pct','sb_pct','dp_pct', 'sn_pct']


# ---------------------------------------------------
# Multiprocessing driver
# ---------------------------------------------------
if __name__ == "__main__":
    start = time.time()

    try:
        with Pool(processes=4) as pool:
            pool.map(create_qc_map, field_list)

        logger.info("QC maps generated successfully.")

    except Exception as e:
        logger.error(f"Multiprocessing failed: {e}", exc_info=True)

    end = time.time()
    logger.info(f"Total time: {round((end - start)/60, 2)} minutes")

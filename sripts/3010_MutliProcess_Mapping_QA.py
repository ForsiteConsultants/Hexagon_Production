import arcpy, os
import time
from multiprocessing import Pool
from shared.logger_utils import get_logger

logger = get_logger('3010_MultiProcess_Mapping_QA')

#######################
## note olga: for canfor GP i ran it as 3 distinct areas
## if you think it makes more sense to do the map as one for the whole TFL, you arem ore than welcome to merge them into one layer
## otherwise, make a folder for each of the areas, and run this bit of code 3 times. Maybe easiest to QA on area C, it's the smallest
#####################

#this piece of code loops throught the dictionary of colours and then activates the field, and applies the colour scale

def create_QC_maps(area, col, aprx_file, field):
    try:
        logger.info(f"Processing field: {field}")
        aprx = arcpy.mp.ArcGISProject(aprx_file)
        m = aprx.listMaps(area)[0]
        lyt = aprx.listLayouts('Area_'+ area)[0]
        lyr = m.listLayers(area + '_hexagon_merged')[0]
        sym = lyr.symbology
        sym.renderer.field = field
        sym.renderer.colorRamp = aprx.listColorRamps(col)[0]
        lyr.symbology = sym
        legend = lyt.listElements("LEGEND_ELEMENT", "Legend")[0]
        legend.addItem(lyr)
        try:
            os.makedirs(os.path.join(output_root, area, "jpeg"))
        except Exception as e:
            logger.warning(f"Could not create jpeg folder: {e}")
        lyt.exportToJPEG(os.path.join(output_root, area, "jpeg", field+".jpg"), resolution=100)
        logger.info(f"Exported JPEG for field: {field}")
        del aprx
    except Exception as e:
        logger.error(f"Error processing field {field}: {e}", exc_info=True)

Start = time.time()

area = 'AMI_AREA_G'
col = 'Voxel Sequential'
field = 'TOTAL_BA_HA'
aprx_file = r"S:\1845\5\03_MappingAnalysisData\01_ArcMapProjects\AreaG_ITI_QC_FINAL_numeric_template.aprx"
output_root = r'S:\1845\5\03_MappingAnalysisData\04_Plotfiles\QC_HEX'

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

# field_list = []

# field_list = ['CON_BA_HA']

args = [(area, col, aprx_file, field) for field in field_list]


if __name__ == '__main__':
    try:
        with Pool(processes=2) as pool:
            pool.starmap(create_QC_maps, args)
        logger.info("Multi-processing completed successfully.")
    except Exception as e:
        logger.error(f"Multi-processing failed: {e}", exc_info=True)

    End = time.time()
    logger.info(f"{round((End - Start)/60, 2)} mins to finish")



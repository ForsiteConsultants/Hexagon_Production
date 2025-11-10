import arcpy
import os
from shared.trees_to_csv_sp_split_ami import *
import time
from shared.logger_utils import get_logger

logger = get_logger('3008_split_hex_by_FMU')

# --- CONFIG
yml_file = r'S:\1845\5\03_MappingAnalysisData\03_Scripts\05_HEXAGON_PRODUCTION\config.yml'
config = read_yaml_config(yml_file)
area              = config['area']
hex_output_folder = config['hex_output_folder']
hex_orig_folder = config['hex_orig_folder']
hex_gdb           = config['hex_gdb']
hex_fc            = config['hex_fc']
merged_fc         = os.path.join(hex_orig_folder, hex_gdb, hex_fc + '_merged')

# --- ENVIRONMENT
arcpy.env.overwriteOutput = True
arcpy.env.outputCoordinateSystem = arcpy.SpatialReference("NAD 1983 UTM Zone 11N")

# --- START TIMER
Start = time.time()

# --- COLLECT UNIQUE, NON‐EMPTY FMU VALUES
fmu_values = set()
with arcpy.da.SearchCursor(merged_fc, ['FMU']) as cursor:
    for (fmu,) in cursor:
        if fmu and str(fmu).strip().upper() != 'NONE':
            fmu_values.add(fmu)

# --- PROCESS EACH FMU

for fmu in sorted(fmu_values):
    try:
        logger.info(f"Processing FMU: {fmu}")
        gdb_name = f"{area}_{fmu}_HEX_Inventory.gdb"
        gdb_path = os.path.join(hex_output_folder, gdb_name)
        if not arcpy.Exists(gdb_path):
            arcpy.CreateFileGDB_management(hex_output_folder, gdb_name)
            logger.info(f"Created GDB: {gdb_name}")

        out_fc = os.path.join(gdb_path, f"AMI_{area}_{fmu}_Hexagon_Inventory")
        where = f"FMU = '{fmu}'"
        if arcpy.Exists(out_fc):
            arcpy.Delete_management(out_fc)
            logger.info(f"Deleted existing feature class: {out_fc}")
        arcpy.Select_analysis(merged_fc, out_fc, where)
        logger.info(f"Selected FMU {fmu} to {out_fc}")

        field_names = [f.name for f in arcpy.ListFields(out_fc)]
        if 'HEXID' in field_names:
            arcpy.AlterField_management(out_fc, 'HEXID', 'HEX_ID')
            logger.info(f"Altered field HEXID to HEX_ID in {out_fc}")

        logger.info(f"Finished FMU: {fmu}")
    except Exception as e:
        logger.error(f"Error processing FMU {fmu}: {e}", exc_info=True)

# --- REPORT TIME
End = time.time()
logger.info(f"Processing completed in {round((End - Start)/60, 2)} minutes")
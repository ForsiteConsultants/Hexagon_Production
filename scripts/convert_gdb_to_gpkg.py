from osgeo import ogr
import os

gdb_path = r"S:\1845\6\03_MappingAnalysisData\02_Data\06_Hexagon_Production\02_Process\hex_orig\AMI_AREA_H_Hexagon.gdb"
layer_name = "AMI_AREA_H_Hexagon_merged"
output_gpkg = r"S:\1845\6\03_MappingAnalysisData\02_Data\06_Hexagon_Production\02_Process\hex_orig\AMI_AREA_H_Hexagon_merged.gpkg"

# Open the GDB
gdb = ogr.Open(gdb_path)
layer = gdb.GetLayerByName(layer_name)

if layer is None:
    raise ValueError(f"Layer '{layer_name}' not found in GDB.")

# Create (or overwrite) GPKG
driver = ogr.GetDriverByName("GPKG")

# Overwrite existing file
if os.path.exists(output_gpkg):
    driver.DeleteDataSource(output_gpkg)

gpkg = driver.CreateDataSource(output_gpkg)

# Copy the single layer
gpkg.CopyLayer(layer, layer_name)

# Cleanup
gpkg = None
gdb = None

print("Layer conversion complete!")

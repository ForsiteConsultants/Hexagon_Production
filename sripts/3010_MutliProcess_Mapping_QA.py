import arcpy, os
import time
from multiprocessing import Pool

#######################
## note olga: for canfor GP i ran it as 3 distinct areas
## if you think it makes more sense to do the map as one for the whole TFL, you arem ore than welcome to merge them into one layer
## otherwise, make a folder for each of the areas, and run this bit of code 3 times. Maybe easiest to QA on area C, it's the smallest
#####################

#this piece of code loops throught the dictionary of colours and then activates the field, and applies the colour scale

def create_QC_maps(area, col, aprx_file, field):
    aprx = arcpy.mp.ArcGISProject(aprx_file)
    #here you use the list, to get the MAP that you want to edit
    m = aprx.listMaps(area)[0]
    #this is the LAYOUT
    lyt = aprx.listLayouts('Area_'+ area)[0]
    # this will get the layer you want from your map (m)
    lyr = m.listLayers(area + '_hexagon')[0]
    # print(lyr.name)
    print(field)
    #This is all sybmology stuff. you'll want to change this section to activate a layer
    sym = lyr.symbology
    sym.renderer.field = field
    
    sym.renderer.colorRamp = aprx.listColorRamps(col)[0]
    lyr.symbology = sym

    legend = lyt.listElements("LEGEND_ELEMENT", "Legend")[0]
    legend.addItem(lyr)

    try:
        os.makedirs(os.path.join(output_root, area, "jpeg"))
    except:
        pass
    lyt.exportToJPEG(os.path.join(output_root, area, "jpeg", field+".jpg"), resolution=200)
    
    del aprx

Start = time.time()

area = 'AMI_AREA_B'
col = 'Voxel Sequential'
field = 'TOTAL_BA_HA'
aprx_file = r"S:\1845\2\03_MappingAnalysisData\01_ArcMapProjects\AreaB_ITI_QC_FINAL_numeric_template.aprx"
output_root = r'S:\1845\2\03_MappingAnalysisData\04_Plotfiles\QC_HEX'

field_list = ['CON_DWB_FACTOR', 'DEC_DWB_FACTOR', 'TOP_HEIGHT', 'MAX_HT_ITI',  
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

field_list = ['Crown_Closure', 'CON_AV_DIAM', 'DEC_AV_DIAM', 'CON_LOREY_HT', 'DEC_LOREY_HT']

args = [(area, col, aprx_file, field) for field in field_list]

if __name__ == '__main__':
    with Pool(processes=2) as pool:
        pool.starmap(create_QC_maps, args)

# create_QC_maps(area, col, aprx_file, field)
End = time.time()

print(round((End - Start)/60, 2), ' mins to finish')



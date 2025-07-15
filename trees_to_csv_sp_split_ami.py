import os
import math
import arcpy
import pickle
from arcpy.sa import *
import yaml
import pandas as pd
from scipy.stats import exponweib
from scipy.optimize import fmin


import numpy as np

arcpy.CheckOutExtension("Spatial")

def read_yaml_config():
	"""
	Read yaml config and return dictionary of items
	"""
	yml_file = r'S:\1845\2\03_MappingAnalysisData\03_Scripts\05_HEXAGON_PRODUCTION\config_hex_B.yml'
	# with open('./data/config/config.yaml', 'r') as file:
	# 	config = yaml.safe_load(file)
	# 	return config
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


def addInfoToTrees(df2, merch_sp = ['sw', 'pl', 'sb', 'fb', 'lt', 'aw', 'bw', 'pb'], plot_multi = 25):
    """
    returns the original tree dataframe with extra fields that have been muliplied by the plot multiplier.
    Keyword arguments:
    df: Full dataframe of all the tree fields
    merch_sp: merch species list, may exclude dead stuff
    plot_multi: plot multiply factor, if hex area = 400, plot_multi = 25, this is the default
    """
    df = df2.copy()
    # PLOT_MULTI is the factor to convert hex sum into per ha (i.e. total m3 to m3/ha)
    df['PLOT_MULTI'] = plot_multi
    df['TOTAL_TREES'] = 1
    df['MERCH_TREES'] = np.where((df.NETMERCHVO > 0) & (df.SPECIES.isin(merch_sp)), 1, 0)
    df['GVOL_ha'] = df['GROSS_VOL'] * df['PLOT_MULTI']
    df['GMVOL_ha'] = df['GROSS_MVOL'] * df['PLOT_MULTI']* df['MERCH_TREES']
    df['NMVOL_ha'] = df['NETMERCHVO'] * df['PLOT_MULTI'] * df['MERCH_TREES']
    df['TOTAL_SPH'] = df.PLOT_MULTI.round()
    df['MERCH_SPH'] = df.MERCH_TREES * df.PLOT_MULTI.round()
    df['BA_ha'] = df.BASAL_AREA * df.PLOT_MULTI
    df['MBA_ha'] = np.where(df.MERCH_TREES > 0, df.BA_ha, 0)

    return df

def topHt(x, pct, col):
    totcount = len(x)
    # print str(totcount) + ' ' + str(pct)
    topcount = int(totcount*pct)
    # print(totcount)
    if topcount < 1:
        topcount = 1
    ht_pct = x.sort_values(col, ascending=False).head(topcount).mean(numeric_only=True)
    # print(ht_pct)
    return ht_pct

def hexHeightInfo(df_orig, hexid = 'HEXID', HexArea = 400):
    """
    Returns a dataframe that has top height, max height, min height, height range and quantiles.
    Keyword arguments:
    df_orig: a dataframe of trees containing the fields hexid,HEIGHT,HEX_AREA_M2, DBH
    """

    df = df_orig.copy()
    # AreaClass = [350, 250, 150, 0]
    i = int(HexArea/100)
    df['HEIGHT'] = df.HEIGHT.astype(float)
    
    df_sub = df.copy()
    df1 = df_sub[[hexid, 'HEIGHT', 'DBH']].groupby(hexid).apply(
        lambda x: x.sort_values('DBH', ascending=False).head(i).mean(numeric_only=True))
    # DOUBLE CHECK THIS WORKS NOW THAT DBH IS THERE
    df1.columns = ['TOP_HEIGHT', 'DBH']
    top_ht = df1[['TOP_HEIGHT']].copy()
    # print(top_ht.head())
    # Height Range -- Used in some Vertical Complexity indexs
    htrng = df[df['GVOL_ha'] > 0].copy().groupby(hexid)[['HEIGHT']].agg(np.ptp)
    htrng.columns = ['Height_Range']

    # min/max ht
    minHt = df[df['GVOL_ha'] > 0].copy().groupby(hexid)[['HEIGHT']].min()
    minHt.columns = ['MinHt']
    maxHt = df.copy().groupby(hexid)[['HEIGHT']].max()
    maxHt.columns = ['MaxHt']
    avgHt = df.copy().groupby(hexid)[['HEIGHT']].mean()
    avgHt.columns = ['AvgHt']

    df_spp = df.copy()
    gb_3 = df.groupby(hexid)[['HEIGHT']].quantile(0.33)
    gb_3.columns = ['QT_3']
    gb_6 = df.groupby(hexid)[['HEIGHT']].quantile(0.66)
    gb_6.columns = ['QT_6']
    gb_q = gb_3.join(gb_6)

    df_spp = pd.merge(df_spp, gb_q.reset_index(), on=hexid)
    df_spp['HT_GRP'] = np.where(df_spp.HEIGHT < df_spp.QT_3, 'S', np.where(df_spp.HEIGHT < df_spp.QT_6, 'M', 'T'))
    df_spp = df_spp.sort_values('BASAL_AREA', ascending=False)

    # add summation here to deterine leading spp in each quantile
    gp_ldsp = pd.crosstab(df_spp[hexid], df_spp.HT_GRP, df_spp.SPECIES, aggfunc='first').join(gb_q).join(maxHt).join(minHt).join(avgHt).join(htrng).join(top_ht)

    return gp_ldsp.reset_index()


def hexSpeciesCrosstabs(df, hexid = 'HEXID'):
    """
    Returns a dataframe that splits up the values by species. either percetns or DWB.
    Keyword arguments:
    df: a dataframe of trees with the multiplier applied
    """
    # all species percentage
    df_spp = df.copy()
    spp = pd.crosstab(df_spp[hexid], df_spp.SPECIES, df_spp.CANOPYAREA, aggfunc=np.sum, normalize='index')
    spp.columns = [x+"_cc_percent" for x in spp.columns]

    # merch species percentage
    df_mspp = df[df['MERCH_TREES'] > 0].copy()
    mspp = pd.crosstab(df_mspp[hexid], df_mspp.SPECIES, df_mspp.CANOPYAREA, aggfunc=np.sum, normalize='index')
    mspp.columns = [x+"_merch_cc_percent" for x in mspp.columns]

    # average DWB factor
    DWB_gb = df[df['MERCH_TREES'] > 0].groupby([hexid, 'SPECIES'], as_index=False)[['GMVOL_ha', 'NMVOL_ha']].sum()
    DWB_gb['DWB_FACTOR'] = 1 - DWB_gb['NMVOL_ha']/DWB_gb['GMVOL_ha']
    DWB_gb['SPECIES'] = DWB_gb.SPECIES + "_DWB"
    DWB_ct = pd.crosstab(DWB_gb[hexid], DWB_gb.SPECIES, DWB_gb.DWB_FACTOR, aggfunc='sum').fillna(0)
    spp_ct = spp.join(mspp).join(DWB_ct)
    return spp_ct.reset_index()

def hexRollupSums(df, hexid='HEXID'):
    """
    this function rolls up SPH (number of trees) of all species (including dead)
    df: ITI data after information added
    """
    Merch_gb = df[df['GVOL_ha'] > 0].copy().groupby(hexid)[['TOTAL_TREES', 'MERCH_TREES']].sum()
    Merch_gb.columns = ['TOTAL_TREES', 'MERCH_TREES']

    return Merch_gb.reset_index()

def hexRollupSumsSpecies(df, hexid='HEXID'):
    """
    roll up per ha information by con/dec
    df: filtered ITI data on con or dec
    """
    all_gb = df[df['GVOL_ha'] > 0].copy().groupby(hexid)[['GVOL_ha', 'GMVOL_ha', 'NMVOL_ha', 'BA_ha', 'MBA_ha', 'TOTAL_TREES', 'MERCH_TREES', 'TOTAL_SPH', 'MERCH_SPH']].sum()
    all_gb.columns = ['GROSS_VOL_HA', 'GROSS_MVOL_HA', 'NETMVOL_HA', 'BA_ha', 'MBA_ha', 'TOTAL_TREES', 'MERCH_TREES', 'TOTAL_SPH', 'MERCH_SPH']

    return all_gb.reset_index()


def hexRollupAverages(df, hexid='HEXID'):
    dbh = df.groupby(hexid)[['LOC_DENSTY', 'AVG_TR_HGT', 'DBH']].mean()
    return dbh.reset_index()

def hexRollupAveragesSpecies(df, hexid='HEXID'):
    # calculate average DBH only on merch trees
    df_m = df[df['MERCH_TREES'] > 0]
    if df_m.empty:
        dbh = df.groupby(hexid)[['DWB_FACTOR']].mean()
    else:
        dbh = df_m.groupby(hexid)[['DWB_FACTOR']].mean()
    return dbh.reset_index()


def hexTallestNTrees(df, n, ot_df, minht, hexid='HEXID'):
    # For trees where top Ht >10 and  keep a list of the first N trees.

    df2 = df.groupby([hexid], as_index=False).apply(lambda x: x.sort_values('HEIGHT', ascending=False))
    df2['ORDER'] = df2.groupby(hexid).cumcount()

    # deleted total trees from being merged in with OT_Df -- may need to readd???
    df3 = df2.merge(ot_df[[hexid, 'TOP_HEIGHT', 'ADJ_TOTAL_SPH','TOTAL_TREES']], on=hexid, how='outer')

    df3['KEEP'] = np.where(df3.TOP_HEIGHT <= minht, 1, np.where(df3.TOTAL_TREES < n, 1, np.where(df3.ADJ_TOTAL_SPH/25 < n, 1, np.where(df3.ORDER < n, 1, 0))))
    df3 = df3[df3.KEEP == 1].reset_index()[[hexid, 'ORDER', 'HEIGHT', 'SPECIES', 'DBH']].copy()
    return df3



def hex_fc_topht_raster(path, Hex_FC, suffix):
    """
    This function convert TOP_HEIGHT from hexagons to raster file

    Input:
    path: gdb path to store temporary output
    HEX_FC: hexagon file that has TOP_HEIGHT attribute
    suffix: this is usually an area or sub-unit

    Output:
    raster file of TOP HEIGHT
    """
    output = os.path.join(path, "TOP_HEIGHT_" + suffix)
    arcpy.conversion.PolygonToRaster(Hex_FC, "TOP_HEIGHT", output, "CELL_CENTER", "NONE", 1)


def cc_calc(path, Hex_FC, CHM, suffix, perc=60, hexid='HEXID'):
    """
    This function calculates crown cover using top height
    Input:
    path: the folder to store rasters and output table, usually a gdb
    
    return:
    a arcgis table contains # of cells gt canopy ht by HEXID
    """
    print("Crown Closure Calc ")
    tpht = Raster(path + r"\TOP_HEIGHT_"+suffix)
    out60pct = tpht * (perc/100)
    out60pct.save(path + r"\pct"+str(perc)+"_"+suffix)
    outCon = Con(Raster(CHM) >= Raster(path + "/pct"+str(perc)+"_"+suffix), 1, 0)
    outCon2 = Con(Raster(CHM) >= 3, outCon, 0)
    outCon2.save(path + "/gt"+str(perc)+"pct_TopHt_"+suffix)
    ZonalStatisticsAsTable(Hex_FC, hexid, path + "/gt"+str(perc)+"pct_TopHt_"+suffix, path+r'\CellContribute_CC_'+str(perc)+"_"+suffix, statistics_type="SUM")
    return path+r'\CellContribute_CC_'+str(perc)+"_"+suffix


def cc_calc_top(path, Hex_FC, CHM, suffix, top_r):
    """
    
    """
    out_top = Divide(path+r"\TOP_HEIGHT_"+suffix, top_r)
    out_top.save(path + r"/Top"+str(top_r)+"m_"+suffix)
    outCon = Con(Raster(CHM) >= Raster(path + r"/Top"+str(top_r)+"m_"+suffix), 1, 0)
    outCon2 = Con(Raster(CHM) >= 3, outCon, 0)
    outCon2.save(path + "/gt"+str(top_r)+"m_TopHt_"+suffix)
    ZonalStatisticsAsTable(Hex_FC, "HEX_ID", path + "/gt"+str(top_r)+"m_TopHt_"+suffix, path+r'\CellContribute_CC_'+str(top_r)+"m_"+suffix, statistics_type="SUM")
    return path+r'\CellContribute_CC_'+str(top_r)+"m_"+suffix


# def dbh(Height, species, nsr, LocD, loc_ht, dbh_params):
#     species = species.upper()
#     if species in ['AW', 'SB', 'SW']:
#         pass
#     else:
#         nsr = 'ALL'

#     if (species, nsr) in dbh_params.keys():
#         b1 = dbh_params[(species, nsr)]['b1']
#         b2 = dbh_params[(species, nsr)]['b2']
#         b3 = dbh_params[(species, nsr)]['b3']
#         b4 = dbh_params[(species, nsr)]['b4']
#         b5 = dbh_params[(species, nsr)]['b5']
#         density = (LocD/1000)**0.5
#         return b1*(Height-1.3)**b2*math.exp(-b3*(Height-1.3))*b4**density*math.exp(b5*loc_ht)
#     else:
#         return -1

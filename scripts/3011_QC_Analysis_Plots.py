import os
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd

# ---------------------------------------------------------
# USER PARAMETERS
# ---------------------------------------------------------
area = 'AREA_H'
GRID_LIST_CSV = multiprocess = r'S:\1845\6\03_MappingAnalysisData\02_Data\06_Hexagon_Production\02_Process\csv_folder\MultiProcessing_files_input_' + area + '.csv'
GDB_FOLDER = r"S:\1845\6\03_MappingAnalysisData\02_Data\06_Hexagon_Production\02_Process\hex_output\GRID"
OUTPUT_FOLDER = r"S:\1845\6\03_MappingAnalysisData\04_Plotfiles\QC_HEX\AMI_AREA_H\var_dist_plots"
N_SAMPLE = 20                                    # number of grids to QC

# Key fields to QC
VOLUME_FIELDS = ['TOTAL_GVOL_PRED_HA','CON_GVOL_PRED_HA', 'DEC_GVOL_PRED_HA',
                 'TOTAL_GMVOL_PRED_HA', 'CON_GMVOL_PRED_HA', 'DEC_GMVOL_PRED_HA',
                'TOTAL_NMVOL_PRED_HA', 'CON_NMVOL_PRED_HA', 'DEC_NMVOL_PRED_HA']

BA_FIELDS = ['TOTAL_BA_HA', 'CON_BA_HA', 'DEC_BA_HA', 
          'TOTAL_MBA_HA', 'CON_MBA_HA', 'DEC_MBA_HA']

SPH_FIELDS = ['TOTAL_SPH_GT_5m', 'CON_SPH_GT_5m', 'DEC_SPH_GT_5m', 
          'TOTAL_MERCH_SPH', 'CON_MERCH_SPH', 'DEC_MERCH_SPH']

HEIGHT_FIELDS = ['CON_LOREY_HT', 'DEC_LOREY_HT', 'TOP_HEIGHT', 'MAX_HT_ITI']

SCATTERS = [
    ("TOP_HEIGHT", "TOTAL_GVOL_PRED_HA"),
    ("DEC_QMD", "DEC_GVOL_PRED_HA"),
    ("CON_QMD", "CON_GVOL_PRED_HA"),
]
# ---------------------------------------------------------


# Make sure seaborn is active
sns.set_theme(style="whitegrid")


def load_grid_list(csv_path):
    df = pd.read_csv(csv_path)
    if "GRID" not in df.columns:
        raise ValueError("CSV must contain a 'GRID' column.")
    return df["GRID"].astype(str).tolist()


def sample_grids(grid_list, n):
    return random.sample(grid_list, min(n, len(grid_list)))


def get_gdb_path(grid_name):
    return os.path.join(GDB_FOLDER, f"HEX_{grid_name}.gdb")


def read_inventory_from_gdb(gdb_path, fc_name):
    fc_path = os.path.join(gdb_path, fc_name)
    try:
        gdf = gpd.read_file(gdb_path, layer=fc_name)
        return gdf
    except Exception as e:
        print(f"Could not load {fc_path}: {e}")
        return None


def ensure_dir(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


# ---------------------------------------------------------
# PLOTTING FUNCTIONS
# ---------------------------------------------------------
def plot_hist(gdf, fields, out_dir, title_prefix):
    for f in fields:
        if f not in gdf.columns:
            continue

        plt.figure(figsize=(8, 6))
        sns.histplot(gdf[f].dropna(), kde=True)
        plt.title(f"{title_prefix} - {f}")
        plt.xlabel(f)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{f}_hist.png"))
        plt.close()


def plot_scatter(gdf, x, y, out_dir):
    if x not in gdf.columns or y not in gdf.columns:
        return

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=gdf[x], y=gdf[y])
    plt.title(f"{x} vs {y}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{x}_vs_{y}.png"))
    plt.close()


# ---------------------------------------------------------
# MAIN QC PROCESS
# ---------------------------------------------------------
def run_qc():
    # Load grid list
    all_grids = load_grid_list(GRID_LIST_CSV)
    sample = sample_grids(all_grids, N_SAMPLE)
    print(f"Selected grids: {sample}")

    for grid in sample:
        print(f"\n=== QC for grid {grid} ===")

        gdb_path = get_gdb_path(grid)
        if not os.path.exists(gdb_path):
            print(f"Missing GDB for {grid}: {gdb_path}")
            continue

        # Read data
        gdf = read_inventory_from_gdb(gdb_path, grid)
        if gdf is None or gdf.empty:
            print(f"No data for grid {grid}")
            continue

        # Create output folder for this grid
        grid_out = os.path.join(OUTPUT_FOLDER, grid)
        ensure_dir(grid_out)

        # Histograms
        plot_hist(gdf, VOLUME_FIELDS, grid_out, f"{grid} Volume")
        plot_hist(gdf, BA_FIELDS, grid_out, f"{grid} Basal Area")
        plot_hist(gdf, SPH_FIELDS, grid_out, f"{grid} SPH")
        plot_hist(gdf, HEIGHT_FIELDS, grid_out, f"{grid} Heights")

        # Scatterplots
        for x, y in SCATTERS:
            plot_scatter(gdf, x, y, grid_out)

        print(f"QC plots saved to {grid_out}")


# ---------------------------------------------------------
# RUN
# ---------------------------------------------------------
if __name__ == "__main__":
    run_qc()

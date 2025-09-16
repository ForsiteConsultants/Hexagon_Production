## Hexagon Production Pipeline

This repository contains scripts and workflows for processing modelling outputs into 4 km hexagon grids, enriching them with attributes, and preparing final outputs for QC and analysis. It is recommended that the scripts be run in the order specified below. Make sure to update the project `config.yml` file.

---

## 1. Grid Processing & Merging Scripts

* **`1001_Data_Prep`**: This script combines all individual hex-grid shapefiles into one master geodatabase layer and sets up a working geodatabase containing a production-ready export grid.
* **`1003_a_multiprocess_ITI_prep`**: This script prepares the input list of ITI feature classes for multiprocessing workflows. It scans a directory of hex-grid shapefiles, and checks which grids have a matching ITI table in the project geodatabase. All valid entries are written to a single CSV file (e.g., `MultiProcessing_files_input_Area_G.csv`) that downstream scripts can use to run batch processing.

## 2. Modeling Outputs

* **`3003_c_MultiProcess_modeling_to_csv`**: Run modeling scripts over hexagon grids.

  * Example outputs:

    ```
    S:/1845/2/03_MappingAnalysisData/02_Data/06_Hexagon_Production/02_Process/csv_output/A_5/A_5_Hex_predicted_output.csv
    S:/1845/2/03_MappingAnalysisData/02_Data/06_Hexagon_Production/02_Process/csv_output/A_5/A_5_treeList_predicted_output.csv
    ```
* **`3003_d_MultiProcess_treelist_to_csv` & `3003_e_MultiProcess_treelist_reformat`** are handled with AWS, the code introduction can be found here: https://forsiteconsultants.sharepoint.com/:o:/s/RMT--Biometrics/Em_Nf7VGdJNDr-z4YbZzLfkBzVivd_FbJNQKSKxBEWUFGg?e=Ndndcv.

## 3. Attribute Enrichment Scripts

1. **`3002_MultiProcess_add_Fields`  & `3004_b_MultiProcess_addFields_treeList`**

   * These scripts add a set of pre-defined data fields (modeling attributes) to each of the hexagon grids. The output is one file geodatabase per grid containing the shapefile with all the required fields added.
   * These steps can be combined into one script, but expect long run times.

2. **`3004_a_MultiProcess_CSV_to_hex` & `3004_c_MultiProcess_treelist_to_hex`**

   * The script **`3004_a_MultiProcess_CSV_to_hex`** populates hexagon attributes using outputs from **ITI complilation** and **Modeling (`3003_c_MultiProcess_modeling_to_csv`)**. 
   * Example output:

     ```
     S:/1845/2/03_MappingAnalysisData/02_Data/06_Hexagon_Production/02_Process/csv_output/A_5/A_5_DEAD_OUTPUT.csv
     ```
   * The script **`3004_c_MultiProcess_treelist_to_hex`** populates hexagon attributes using treelist outputs from **`3003_d_MultiProcess_treelist_to_csv` & `3003_e_MultiProcess_treelist_reformat`**, which are ran on AWS. Therefore this part can be completed later once the AWS output is ready.

## 4. Crown Closure (CC) Calculation

* **`3005_MultiProcess_CC_calc`**: This script parallelizes the computation of crown-closure percentages across all hexagon grids, using LiDAR CHM and each grid’s top-height data, and writes the results back into the geodatabases while keeping a full audit trail.

## 5. Final Edits & Merging

1. **`3006_final_edits_add_NSR` & `3006_final_edits_add_FMU`**

   * Add NSR and FMU attributes to the enriched hexagon dataset.
   * Part 1 runs single-threaded; Part 2 can be parallelized.

2. **`3007_merge_hex_grids`**

   * Merge individual 4 km grids into one comprehensive file for QC.

3. **`3008_split_hex_by_FMU`**

   * Split the merged hexagon dataset by FMU for easier use by the client.

4. **`3010_MultiProcess_Mapping_QA`**

   * QC mapping: review image output quality and optimize for faster rendering.

---



## Contact

For questions or support, contact Anita Li/ Jimmy Ke/ Kwame Awuah.
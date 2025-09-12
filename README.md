## Hexagon Production Pipeline

This repository contains scripts and workflows for processing modelling outputs into 4 km hexagon grids, enriching them with attributes, and preparing final outputs for QC and analysis.

---

## 1. Grid Processing & Merging

* **Process**: Split the study area into 4 km hexagon grids.
* **Merge**: Consolidate all grid outputs into a unified dataset for downstream tasks.

## 2. Attribute Enrichment

1. **3002 & 3004\_b**

   * Add modeling attributes to the hexagon grids.
   * These steps can be combined into one script, but expect long run times.

2. **3004\_a**

   * Populate hexagon attributes using outputs from **ITI complilation** and **3003\_c** (no treelist output needed).
   * Output:

     ```
     S:/1845/2/03_MappingAnalysisData/02_Data/06_Hexagon_Production/02_Process/csv_output/A_5/A_5_DEAD_OUTPUT.csv
     ```

3. **3004\_c**

   * Attach treelist outputs back to hexagon grids.

## 3. ITI Compilation


* Prep multiprocessing files and complete ITI compilation for the full AOI.
* Exclude `admin_fields` from ITI inputs.
* Outputs:

  ```
  S:/1845/2/03_MappingAnalysisData/02_Data/06_Hexagon_Production/02_Process/csv_output/A_5/A_5_ITI_treelist.csv
  S:/1845/2/03_MappingAnalysisData/02_Data/06_Hexagon_Production/02_Process/csv_output/A_5/OUTPUT_SUM_CON_A_5.csv
  S:/1845/2/03_MappingAnalysisData/02_Data/06_Hexagon_Production/02_Process/csv_output/A_5/OUTPUT_SUM_DEC_A_5.csv
  S:/1845/2/03_MappingAnalysisData/02_Data/06_Hexagon_Production/02_Process/csv_output/A_5/OUTPUT_SUM_TOTAL_A_5.csv
  ```

## 4. Modeling Outputs

* **3003\_c**: Run modeling scripts over hexagon grids.

  * Outputs:

    ```
    S:/1845/2/03_MappingAnalysisData/02_Data/06_Hexagon_Production/02_Process/csv_output/A_5/A_5_Hex_predicted_output.csv
    S:/1845/2/03_MappingAnalysisData/02_Data/06_Hexagon_Production/02_Process/csv_output/A_5/A_5_treeList_predicted_output.csv
    ```
*  3003\_d & 3003\_e are handled with AWS, the code introduction can be found here: https://forsiteconsultants.sharepoint.com/:o:/s/RMT--Biometrics/Em_Nf7VGdJNDr-z4YbZzLfkBzVivd_FbJNQKSKxBEWUFGg?e=Ndndcv.

## 5. Crown Closure (CC) Calculation

* **3005**: Calculate CC per grid cell sequentially (no multiprocessing due to file locks).
* Future improvement: consider using GeoPandas or other parallel approaches.

## 6. Final Edits & Merging

1. **3006**

   * Add NSR and FMU attributes to the enriched hexagon dataset.
   * Part 1 runs single-threaded; Part 2 can be parallelized.

2. **3007**

   * Merge individual 4 km grids into one comprehensive file for QC.

3. **3008**

   * Split the merged hexagon dataset by FMU for easier use by the client.

4. **3010**

   * QC mapping: review image output quality and optimize for faster rendering.

---



## Contact

For questions or support, contact Anita Li and Jimmy Ke.

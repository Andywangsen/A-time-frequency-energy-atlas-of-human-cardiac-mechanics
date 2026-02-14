# Code Location Guide

To avoid code duplication and maintain the integrity of the analysis pipeline, the generation code for **Extended Data Fig. 4** is included within the main figure scripts in the `Fig. 3`, `Fig. 4`, and `Fig. 5` directories.

## File Mappings:

*   **Extended Data Fig. 4a**:
    *   **Script Path**: `../../Fig. 3/Fig. 3b/Fig. 3b_ROC_Permutation_Test.py`
    *   **Generated Files**:
        *   Extended Data Fig. 4a_Confusion_Matrix_Independent_Test.png
        *   Extended Data Fig. 4a_Threshold_Tuning_Independent_Test.png

*   **Extended Data Fig. 4b, 4c, 4d**:
    *   **Script Path**: `../../Fig. 4/Fig. 4c,4d/Fig. 4d_External_Validation_ROC.py`
    *   **Generated Files**:
        *   Extended Data Fig. 4b_Confusion_Matrix_External_Validation_CVD vs HF.png
        *   Extended Data Fig. 4c_Confusion_Matrix_External_Validation_CVD-LRRCV vs HF-LRRCV.png
        *   Extended Data Fig. 4d_Confusion_Matrix_External_Validation_CVD-HRRCV vs HF-HRRCV.png

*   **Extended Data Fig. 4e**:
    *   **Script Path**: `../../Fig. 5/Fig. 5e,5f/Fig. 5f_HF_Phenotyping_External_Val.py`
    *   **Generated Files**:
        *   Extended Data Fig. 4e_Confusion_Matrix_External_Validation_HFpEF vs HFrEF.png
        *   Extended Data Fig. 4e_Threshold_Tuning_External_HFpEF vs HFrEF.png

Please run the corresponding scripts listed above to generate the specific charts for Extended Data Fig. 4.

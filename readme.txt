Research pipeline for predicting per-walk MDS-UPDRS gait scores from nine-joint, SMPL-derived 3D skeleton sequences. The current stage is data conversion and validation; no CST-GCN results are reported yet.


Data and scope

This project builds on CARE-PD. Its SMPL parameter files (.pkl) are converted to skeleton arrays with shape (T, 9, 3). Cohorts with a valid UPDRS_GAIT field supply a per-walk score; walks without that field remain unlabeled. Obtain the data and the SMPL model separately using the CARE-PD project's instructions. This repository does not redistribute clinical recordings or SMPL model assets.


Steps 

1. Download SMPL Files fron dataverse for each cohort as .pkl fles
2. Run the preprocessing code to convert to smpl9 convention which is the format used for CST-GCN 
	 python .\data\preprocessing\smpl2smpl9_world_with_labels.py --all
	The preprocessed datasets are stored in assets\datasets\processed\smpl9 with the respective cohort dataset name similar to pkl
3.
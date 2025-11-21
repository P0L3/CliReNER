A file that notes the used commands during the experiment.

**create_hf_dataset.py**

Command used to create CliReNER SILVER on HF:
```shell
python3 -m EXPERIMENTS.create_hf_dataset --lsfile_path "RESULTS/training_50_EneSou_BodOfWat_Org_PhyPhe_Loc_PhyArt_NatDis_Che_BodPar_MatExp_GeoFea_Org_IntArt_Sys_Ass_Met_FieOfStu_MetPhe_TimPer_Eco_Pol_NatPhe_Qua_Per_Dis_MeaDev_Sat.json" --hf_name "P0L3/CliReNER_v_1_1_28_SILVER"
```
Command used to create CliReNER GOLD on HF (preannotator annotations):
```shell
python3 -m EXPERIMENTS.create_hf_dataset --lsfile_path "RESULTS/golden_50_EneSou_BodOfWat_Org_PhyPhe_Loc_PhyArt_NatDis_Che_BodPar_MatExp_GeoFea_Org_IntArt_Sys_Ass_Met_FieOfStu_MetPhe_TimPer_Eco_Pol_NatPhe_Qua_Per_Dis_MeaDev_Sat.json" --hf_name "P0L3/CliReNER_v_1_1_28_GOLD_authorannots"
```


**calculate_hf_dataset_stats.py**

Command used to output statistics for CliReNER SILVER:
```shell
python3 EXPERIMENTS/calculate_hf_dataset_stats.py --dataset "P0L3/CliReNER_v_1_1_28_SILVER" --output_dir "EXPERIMENTS/DATASET_STATS"
```

Command used to output statistics for CliReNER GOLD (preannotator annotations):
```shell
python3 EXPERIMENTS/calculate_hf_dataset_stats.py --dataset "P0L3/CliReNER_v_1_1_28_GOLD_authorannots" --output_dir "EXPERIMENTS/DATASET_STATS"
```
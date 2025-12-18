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


**finetune.py**

Command used for initial experiments with model fine-tuning (GLiNER):
```shell
python3 -m EXPERIMENTS.finetune --model_type GLINER --dataset_id P0L3/CliReNER_v_1_1_28_SILVER --model_id gliner-community/gliner_medium-v2.5 --config_path EXPERIMENTS/gliner_config.json
```

COmmand used when wnadb was implementetd (GLiNER):
```shell
python -m EXPERIMENTS.finetune --model_type GLINER --dataset_id P0L3/CliReNER_v_1_1_28_SILVER --model_id gliner-community/gliner_medium-v2.5 --config_path EXPERIMENTS/gliner_config.json --wandb_project "CLIRENER_SILVER_EXPERIMENTS" --wandb_name "GLiNER_Medium_v2.5_Run"
```

**evaluate.py**

Command used when wandb was implemented; evaluation (GLiNER): 27.11.2025.
```shell
python -m EXPERIMENTS.evaluate  --model_type GLINER  --dataset_id P0L3/CliReNER_v_1_1_28_SILVER  --model_path EXPERIMENTS/models/GLINER/gliner_medium_v2_5_CliReNER_v_1_1_28_SILVER/checkpoint-final  --wandb_project "CLIRENER_SILVER_EXPERIMENTS"  --wandb_run_id sd7yuocj
```

Command used with both implemented and joined in pipiline (GLiNER): 27.11.2025.
```shell
python -m EXPERIMENTS.run_pipeline  --model_type GLINER  --dataset_id P0L3/CliReNER_v_1_1_28_SILVER  --model_id gliner-community/gliner_medium-v2.5  --config_path EXPERIMENTS/gliner_config.json  --wandb_project "CLIRENER_SILVER_EXPERIMENTS"  --wandb_name "GLiNER_Medium_v2.5_Pipeline_Run"
```

Command used when wandb was implemented and updated with visuals; evaluation (GLiNER): 27.11.2025.
```shell
python -m EXPERIMENTS.evaluate  --model_type GLINER  --dataset_id P0L3/CliReNER_v_1_1_28_SILVER  --model_path FINETUNES/GLINER/models/GLINER_med_v2_5/checkpoint-final  --wandb_project "CLIRENER_SILVER_EXPERIMENTS"
```

**finetune_evaluate_pipeline.py**

Command used with both implemented and joined in pipiline (SPANMARKER): 27.11.2025.
```shell
python -m EXPERIMENTS.finetune_evaluate_pipeline  --model_type SPANMARKER  --dataset_id P0L3/CliReNER_v_1_1_28_SILVER  --model_id P0L3/clirebert_clirevocab_uncased  --config_path EXPERIMENTS/spanmarker_config.json  --wandb_project "CLIRENER_SILVER_EXPERIMENTS"  --wandb_name "SpanMarker_CliReBert_Pipeline_Run"
```


**gold_evaluate.py**

Command used to run evaluation on GOLD dataset with only author annotations:
```shell
python -m EXPERIMENTS.gold_evaluate
```

**zeroshot_evaluate.py**
Command used to evaluate GLiNER with no fine-tuning:
```shell
python -m EXPERIMENTS.zeroshot_evaluate
```


**run_seeds_(gliner|spanmarker).sh**
Command used to run full training and evaluation script on Windows PC1 and/or PC2:
```
"C:\Program Files\Git\bin\sh.exe" ./EXPERIMENTS/run_seeds_spanmarker.sh  
```
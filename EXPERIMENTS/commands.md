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

Command used to create IBMCCNER annotated by CliReNER schema (no test and val):
```shell
python3 -m EXPERIMENTS.create_hf_dataset --lsfile_path "DATA/LABEL_STUDIO/LS_IBMCCNER/project-36-at-2025-12-08-13-43-e4ba28f1.json" --hf_name "P0L3/Climate-Change-NER-S50-CliReNER" --test_s 0.0 --val_s 0.0
```

Command used to create BioDivNER annotated by CliReNER schema (no test and val):
```shell
python3 -m EXPERIMENTS.create_hf_dataset --lsfile_path "DATA/LABEL_STUDIO/LS_BIODIVNER/project-34-at-2025-12-08-13-44-2171ef2d.json" --hf_name "P0L3/BiodivNER-S50-CliReNER" --test_s 0.0 --val_s 0.0
```

Command used to create ClimateIE annotated by CliReNER schema (no test and val):
```shell
python3 -m EXPERIMENTS.create_hf_dataset --lsfile_path "DATA/LABEL_STUDIO/LS_CLIMATEIE/project-35-at-2025-12-05-07-53-b0139f8e.json" --hf_name "P0L3/ClimateIE-S50-CliReNER" --test_s 0.0 --val_s 0.0
```

Command used to create CliReNER GOLD on HF (aggregated annotations):
```
python3 -m EXPERIMENTS.create_hf_dataset --lsfile_path "/home/p0l3/RAD/DROP/CLIRENER/ANNOTATORS/CONSENSUS/6326.json" --hf_name "P0L3/CliReNER_v_1_1_28_GOLD"
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

Command used to output statistics for IBMCCNER mapped to CliReNER schema:
```shell
python3 EXPERIMENTS/calculate_hf_dataset_stats.py --dataset "P0L3/Climate-Change-NER-S50-CliReNER" --output_dir "EXPERIMENTS/DATASET_STATS"
```

Command used to output statistics for BioDivNER mapped to CliReNER schema:
```shell
python3 EXPERIMENTS/calculate_hf_dataset_stats.py --dataset "P0L3/BiodivNER-S50-CliReNER" --output_dir "EXPERIMENTS/DATASET_STATS"
```

Command used to output statistics for ClimateIE mapped to CliReNER schema:
```shell
python3 EXPERIMENTS/calculate_hf_dataset_stats.py --dataset "P0L3/ClimateIE-S50-CliReNER" --output_dir "EXPERIMENTS/DATASET_STATS"
```

Command used to output statistics for CliReNER GOLD:
```shell
python3 EXPERIMENTS/calculate_hf_dataset_stats.py --dataset "P0L3/CliReNER_v_1_1_28_GOLD" --output_dir "EXPERIMENTS/DATASET_STATS"
```

**compare_hf_datasets_stats.py**

Command to compare SILVER and GOLD (author annotated) CliReNER dataset:
```shell
python3 EXPERIMENTS/compare_hf_datasets_stats.py
```

Command to compare CLimateIE mapped to CliReNER schema and BioDivNER mapped to CliReNER schema datasets:
```shell
python3 -m EXPERIMENTS.compare_hf_datasets_stats --d1 P0L3/ClimateIE-S50-CliReNER --d2 P0L3/BiodivNER-S50-CliReNER
```

Command to compare SILVER and GOLD CliReNER dataset:
```shell
python3 -m EXPERIMENTS.compare_hf_datasets_stats --d1 "P0L3/CliReNER_v_1_1_28_SILVER" --d2 "P0L3/CliReNER_v_1_1_28_GOLD" 
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
python -m EXPERIMENTS.evaluate_gold
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

**generate_comparative_table.py**
Command used to run qualitative analysis on fine-tuned models:
```
python -m EXPERIMENTS.generate_comparative_table --baseline_id "distilbert/distilroberta-base" --challenger_id "P0L3/sciclimatebert" --baseline_type SPANMARKER --challenger_type SPANMARKER
```
or this command for all possible combinations in encoder families:
```
python -m EXPERIMENTS.generate_comparative_table_grouprun
```
#!/usr/bin/env bash
while ! grep -q 'config_DTU2_Bonly_C_54000.json' 3_processed_configs.json;
do
    run_lsf lsf/scripts/run_inference_3.sh;
done
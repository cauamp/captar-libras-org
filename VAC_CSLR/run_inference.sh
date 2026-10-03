#!/bin/bash

# Verifica se foi fornecido um parâmetro de escolha de script
if [ $# -lt 1 ]; then
    echo "Por favor, forneça o número do script a ser executado (0 ou 1) e opcionalmente o valor da variável input."
    exit 1
fi

# Parâmetros
script_choice=$1

# Verifica qual script deve ser executado com base no parâmetro fornecido
if [ $script_choice -eq 0 ]; then
    input_value=${2:-./preprocess/out/18-09-2024_17-17-33_p0999_s253_b4_RAW--ZED-RGB_FRONT.mp4}
    echo "Executando inferência..."
    python3 inference_one_video.py --load-weights ./work_dir/captar-libras_16-10_b_c_il_int-npz_cam-sim_2/dev_08.25_epoch80_model.pt --phase test --config ./configs/captar_inference.yaml --input $input_value


elif [ $script_choice -eq 1 ]; then
    weights_path=${2:-./work_dir/captar-libras_16-10_in_ctrl/ctrl_73.81dev_68.26_epoch30_model.pt}
    
    echo "Executando eval_inference"
    python3 eval_inference.py --load-weights $weights_path --phase test --config ./configs/captar_eval_inference.yaml

else
    echo "Escolha inválida. Por favor, escolha 0 ou 1."
fi
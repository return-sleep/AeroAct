model_name_or_path=$1
inference_sample_temperature=$2

python eval.py --model_path $model_name_or_path --inference_sample_temperature $inference_sample_temperature 

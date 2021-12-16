export TRANSFORMERS_CACHE=/userhome/.cache
export TOKENIZERS_PARALLELISM=true
python run_mlm.py \
    --model_name_or_path bert-base-multilingual-cased \
    --train_file ./data/shufed_wiki_subset_en_de.txt \
    --validation_file ./data/temp.txt \
    --save_steps 1 \
    --do_train \
    --do_eval \
    --ignore_data_skip \
    --output_dir ./output

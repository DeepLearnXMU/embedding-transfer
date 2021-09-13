python setup.py build_ext --inplace
python train.py \
    data-bin/no_pretrained_wmt14_en_de \
    --restore-file /userhome/lx_2021/pretrained_models_exp/self_attention_transformers/examples/language-modeling/output/checkpoint-12000 \
    --arch transformer_bert2bert --share-decoder-input-output-embed \
    --optimizer adafactor --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 \
    --fp16 \
    --memory-efficient-fp16 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric

fairseq-generate data-bin/no_pretrained_wmt14_en_de \
    --path checkpoints/checkpoint_best.pt \
    --results-path result \
    --batch-size 32 --beam 4 --remove-bpe 

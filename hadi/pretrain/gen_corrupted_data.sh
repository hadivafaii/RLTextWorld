for pretrain_mode in 'ACT_VERB' 'ACT_ENTITY' 'OBS_VERB' 'OBS_ENTITY'
do
  echo "getting pretrain data $pretrain_mode"
  screen -dmS "get_corrupted_$pretrain_mode"
  screen -S "get_corrupted_$pretrain_mode" -X stuff "python3 gen_pretrain_data.py tw_cooking/train $pretrain_mode --mask_prob 0.5 --seeds 110 121 332 443 554 665 ^M"
done

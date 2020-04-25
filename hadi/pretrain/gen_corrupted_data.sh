game_type="tw_cooking/train"

for pretrain_mode in 'ACT_VERB' 'ACT_ENTITY' 'OBS_VERB' 'OBS_ENTITY'
do
  echo "getting pretrain data $pretrain_mode"
  screen -dmS "get_corrupted_$pretrain_mode"
  screen -S "get_corrupted_$pretrain_mode" -X stuff "python3 gen_pretrain_data.py $game_type $pretrain_mode --mask_prob 0.3 --seeds 110 121 332 443 554 665 ^M"
done

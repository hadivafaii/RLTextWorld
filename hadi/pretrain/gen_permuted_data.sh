game_type="tw_cooking/train"

for pretrain_mode in 'ACT_ORDER' 'OBS_ORDER'
do
  echo "getting pretrain data $pretrain_mode"
  screen -dmS "get_permuted_$pretrain_mode"
  screen -S "get_permuted_$pretrain_mode" -X stuff "python3 gen_pretrain_data.py $game_type $pretrain_mode --k 3 ^M"
done

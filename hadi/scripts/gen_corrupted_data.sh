# game_type e.g. tw_cooking/train
game_type=$1
game_spec=$2

# get correct exploration mode
if [[ $game_type =~ "tw_cooking" ]]; then
   game_spec=""
fi

IFS="/"
read -ra ADDR <<< "$game_type"
IFS=" "

# where python scripts are at
cd ../utils

for pretrain_mode in 'ACT_VERB' 'ACT_ENTITY' 'OBS_VERB' 'OBS_ENTITY'
do
  echo "getting pretrain data $pretrain_mode"
  screen -dmS "get_corrupted_${ADDR[0]}_${ADDR[1]}_$pretrain_mode"
  if [ -z "$game_spec" ]; then # if game_spec is NULL
    screen -S "get_corrupted_${ADDR[0]}_${ADDR[1]}_$pretrain_mode" -X stuff "python3 gen_pretrain_data.py $game_type $pretrain_mode --mask_prob 0.3 --seeds 110 121 332 443 554 665 ^M"
  else
    screen -S "get_corrupted_${ADDR[0]}_${ADDR[1]}_$pretrain_mode" -X stuff "python3 gen_pretrain_data.py $game_type $pretrain_mode --mask_prob 0.3 --seeds 110 121 332 443 554 665 --game_spec $game_spec ^M"
  fi
done

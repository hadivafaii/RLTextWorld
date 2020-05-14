# game_type e.g. tw_cooking/train
game_type=$1
game_spec=$2
mask_prob=$3

# get correct exploration mode
if [[ $game_type =~ "tw_cooking" ]]; then
   game_spec=""
fi

IFS="/"
read -ra ADDR <<< "$game_type"
IFS=" "

# where python scripts are at
cd ../utils

for pretrain_mode in 'ACT_VERB' 'ACT_ENTITY' 'OBS_VERB' 'OBS_ENTITY' 'MLM'
do
  echo "getting pretrain data $game_type mask_prob = $mask_prob $pretrain_mode"
  screen -dmS "get_corrupted_${ADDR[0]}_${ADDR[1]}_$pretrain_mode"
  if [ -z "$game_spec" ]; then # if game_spec is NULL
    screen -S "get_corrupted_${ADDR[0]}_${ADDR[1]}_$pretrain_mode" -X stuff "python3 gen_pretrain_data.py $game_type $pretrain_mode --mask_prob $mask_prob --seeds 110 121 332 443 554 665 776 887 998 1110 ^M"
  else
    screen -S "get_corrupted_${ADDR[0]}_${ADDR[1]}_$pretrain_mode" -X stuff "python3 gen_pretrain_data.py $game_type $pretrain_mode --mask_prob $mask_prob --seeds 110 121 332 443 554 665 776 887 998 1110 --game_spec $game_spec ^M"
  fi
done

# manual
# python3 gen_pretrain_data.py custom/train MLM --mask_prob 0.15 --seeds 110 121 332 443 554 665 776 887 998 1110 1221 1332 1443 1554 1665 1776 1887 1998 2110 2221 2332 2443 2554 2665 2776 2887 2998 --game_spec b-simple

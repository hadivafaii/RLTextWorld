# game_type e.g. tw_cooking/train
game_type=$1
game_spec=$2
k=$3

# get correct exploration mode
if [[ $game_type =~ "tw_cooking" ]]; then
   game_spec=""
fi

IFS="/"
read -ra ADDR <<< "$game_type"
IFS=" "

# where python scripts are at
cd ../utils

for pretrain_mode in 'ACT_ORDER' 'OBS_ORDER'
do
  echo "getting pretrain data $game_type $pretrain_mode k = $k"
  screen -dmS "get_permuted_${ADDR[0]}_${ADDR[1]}_$pretrain_mode"
  if [ -z "$game_spec" ]; then # if game_spec is NULL
    screen -S "get_permuted_${ADDR[0]}_${ADDR[1]}_$pretrain_mode" -X stuff "python3 gen_pretrain_data.py $game_type $pretrain_mode --k $k ^M"
  else
    screen -S "get_permuted_${ADDR[0]}_${ADDR[1]}_$pretrain_mode" -X stuff "python3 gen_pretrain_data.py $game_type $pretrain_mode --k $k --game_spec $game_spec ^M"
  fi
done

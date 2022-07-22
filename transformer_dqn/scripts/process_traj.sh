# game_type e.g. tw_cooking/train
# game type is string, eg: tw_cooking_train. num_groups is integer larger than 0. e.g.: 20
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

for ss in 384 512 768 1024; do
  screen -dmS "process_raw_${ADDR[0]}_${ADDR[1]}_${game_spec}_${ss}" &&
  if [ -z "$game_spec" ]; then # if game_spec is NULL
    screen -S "process_raw_${ADDR[0]}_${ADDR[1]}_${game_spec}_${ss}" -X stuff "python3 process_traj.py $game_type $ss ^M"
  else
    screen -S "process_raw_${ADDR[0]}_${ADDR[1]}_${game_spec}_${ss}" -X stuff "python3 process_traj.py $game_type $ss --game_spec $game_spec ^M"
  fi
done

echo "done!"

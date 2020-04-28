# game_type e.g. tw_cooking/train
game_type=$1

# get correct exploration mode
if [[ $game_type =~ "tw_cooking" ]]; then
   game_specs=""

 elif [[ $game_type =~ "tw_simple" ]]; then
   exploration_mode=policy
   echo enter goal and rewards values
   read goal rewards
   game_specs="goal=$goal-rewards=$rewards"

 elif [[ $game_type =~ "custom" ]]; then
   exploration_mode=policy
   echo enter goal, world size, nb objects, and quest length
   read goal wsz nbobj qlen
   game_specs="$goal/wsz=$wsz-nbobj=$nbobj-qlen=$qlen"
fi

echo "loading from .../$game_type/$game_specs"



IFS="/"
read -ra ADDR <<< "$game_type"
IFS=" "

for ss in 384 512 768 1024 2048
do
  screen -dmS "process_raw_${ADDR[0]}_${ADDR[1]}_$ss" &&
  if [ -z "$game_specs" ]; then # if load dir is NULL
    screen -S "process_raw_${ADDR[0]}_${ADDR[1]}_$ss" -X stuff "python3 preprocessing.py $game_type $ss ^M"
  else
    screen -S "process_raw_${ADDR[0]}_${ADDR[1]}_$ss" -X stuff "python3 preprocessing.py $game_type $ss --game_specs $game_specs ^M"
  fi
done

echo "done!"

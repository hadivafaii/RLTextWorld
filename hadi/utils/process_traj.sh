# game_type e.g. tw_cooking/train
# game type is string, eg: tw_cooking_train. num_groups is integer larger than 0. e.g.: 20
game_type=$1

# get correct exploration mode
if [[ $game_type =~ "tw_cooking" ]]; then
   game_specs=""

 elif [[ $game_type =~ "tw_simple" ]]; then
   echo "enter goal {detailed, brief, none} and rewards {dense,balanced,sparse} values"
   read goal rewards
   game_specs="goal=$goal-rewards=$rewards"

 elif [[ $game_type =~ "custom" ]]; then
   echo "enter goal {detailed, brief} game spec (e.g. small large etc)"
   read goal spec
   game_specs="/$goal/$spec"
 else echo "wrong game type entered. exiting..."
   exit 1
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

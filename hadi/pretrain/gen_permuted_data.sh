# game_type e.g. tw_cooking/train
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



for pretrain_mode in 'ACT_ORDER' 'OBS_ORDER'
do
  echo "getting pretrain data $pretrain_mode"
  screen -dmS "get_permuted_$pretrain_mode"
  screen -S "get_permuted_$pretrain_mode" -X stuff "python3 gen_pretrain_data.py $game_type $pretrain_mode --k 3 --game_specs $game_specs ^M"
done

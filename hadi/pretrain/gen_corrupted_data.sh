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
fi

echo "loading from .../$game_type/$game_specs"


for pretrain_mode in 'ACT_VERB' 'ACT_ENTITY' 'OBS_VERB' 'OBS_ENTITY'
do
  echo "getting pretrain data $pretrain_mode"
  screen -dmS "get_corrupted_$pretrain_mode"
  screen -S "get_corrupted_$pretrain_mode" -X stuff "python3 gen_pretrain_data.py $game_type $pretrain_mode --mask_prob 0.3 --seeds 110 121 332 443 554 665  --game_specs $game_specs ^M"
done

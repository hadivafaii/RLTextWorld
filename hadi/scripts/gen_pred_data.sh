### Enter parameters
# game type is string, eg: tw_cooking_train. num_groups is integer larger than 0. e.g.: 20
game_type=$1
num_groups=$2
game_spec=$3

# get correct exploration mode
if [[ $game_type =~ "tw_cooking" ]]; then
   exploration_mode=walkthrough
   game_specs=""
 elif [[ $game_type =~ "tw_simple" || $game_type =~ "custom" ]]; then
   exploration_mode=policy
 else echo "wrong game type entered. exiting..."
   exit 1
fi

# where python scripts are at
cd ..

make_screens () {
  # $1 is num_groups
  for ii in $(seq 0 1 $(($1-1))); do
    screen -dmS "iter_${ii}"
  done
}

do_it () {
  # $1 is num_groups, $2 game_type, $3 is exploration mode, $4 is game spec
  for ii in $(seq 0 1 $(($2-1))); do
    if [ $# -eq 3 ]; then # game spec was given
      screen -S "iter_${ii}" -X stuff "python3 -m utils.gen_pred_data $1 $ii $2 --game_spec $3 ^M"
    else
      screen -S "iter_${ii}" -X stuff "python3 -m utils.gen_pred_data $1 $ii $2 ^M"
    fi
  done
}


make_screens $num_groups


if [ -z "$game_spec" ]; then
  do_it $game_type $num_groups
else
  do_it $game_type $num_groups $game_spec
fi

echo "gen pred jobs created and sent to screens"

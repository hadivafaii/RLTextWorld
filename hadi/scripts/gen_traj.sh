### Enter parameters
# game type is string, eg: tw_cooking_train. num_groups is integer larger than 0. e.g.: 20
game_type=$1
game_spec=$2
num_groups=$3


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
cd ../utils

make_screens () {
  # $1 is num_groups
  for eps in $(seq 0.00 0.20 1.00); do
    for ii in $(seq 0 1 $(($1-1))); do
      screen -dmS "iter_${ii}-eps_${eps}"
    done
  done
}

do_it () {
  # $1 is num_groups, $2 game_type, $3 is exploration mode, $4 is load_dir
  for eps in $(seq 0.00 0.20 1.00); do
    for ii in $(seq 0 1 $(($2-1))); do
      if [ $# -eq 4 ]; then # load dir was given
        screen -S "iter_${ii}-eps_${eps}" -X stuff "python3 gen_traj.py $1 $ii $2 --exploration_mode $3 --game_spec $4 --epsilon $eps --max_steps 70 --batch_size 2 ^M"
      else
        screen -S "iter_${ii}-eps_${eps}" -X stuff "python3 gen_traj.py $1 $ii $2 --exploration_mode $3 --epsilon $eps --max_steps 70 --batch_size 2 ^M"
      fi
    done
  done
}


make_screens $num_groups


if [ -z "$game_spec" ]; then # if load dir is NULL
  do_it $game_type $num_groups $exploration_mode
else
  do_it $game_type $num_groups $exploration_mode $game_spec
fi

echo "jobs created and sent to screens"

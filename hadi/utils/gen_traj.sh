### Enter parameters
# game type is string, eg: tw_cooking_train. num_groups is integer larger than 0. e.g.: 20
game_type=$1
num_groups=$2

# get correct exploration mode
if [[ $game_type =~ "tw_cooking" ]]; then
   exploration_mode=walkthrough
   game_specs=""

 elif [[ $game_type =~ "tw_simple" ]]; then
   exploration_mode=policy
   echo "enter goal {detailed, brief, none} and rewards {dense,balanced,sparse} values"
   read goal rewards
   game_specs="goal=$goal-rewards=$rewards"

 elif [[ $game_type =~ "custom" ]]; then
   exploration_mode=policy
   echo "enter goal {detailed, brief} game spec (e.g. small large etc)"
   read goal spec
   game_specs="/$goal/$spec"
fi

echo "loading from .../$game_type/$game_specs"



make_screens () {
  # $1 is num_groups
  for eps in $(seq 0.00 0.10 1.00); do
    for ii in $(seq 0 1 $(($1-1))); do
      screen -dmS "iter_$ii-eps_$eps"
    done
  done
}

do_it () {
  # $1 is num_groups, $2 game_type, $3 is exploration mode, $4 is load_dir
  for eps in $(seq 0.00 0.10 1.00); do
    for ii in $(seq 0 1 $(($1-1))); do
      if [ $# -eq 4 ]; then # load dir was given
        screen -S "iter_$ii-eps_$eps" -X stuff "python3 gen_traj.py $2 $ii $1 --exploration_mode $3 --game_specs $4 --epsilon $eps --max_steps 70 --extra_episodes 1 --batch_size 2 ^M"
      else
        screen -S "iter_$ii-eps_$eps" -X stuff "python3 gen_traj.py $2 $ii $1 --exploration_mode $3 --epsilon $eps --max_steps 70 --extra_episodes 1 --batch_size 2 ^M"
      fi
    done
  done
}

make_screens $num_groups

if [ -z "$game_specs" ]; then # if load dir is NULL
  do_it $num_groups $game_type $exploration_mode
else
  do_it $num_groups $game_type $exploration_mode $game_specs
fi

echo "jobs created and sent to screens"





# add this line to the end of the command before ^M to also save stdout and stderr in log.txt:
# 2>&1 | tee log-$ii.txt

### kill all screens with epsilon in their names
# screen -ls  | egrep "^\s*[0-9]+.iter_+[0-9.]" | awk -F "." '{print $1}' | xargs kill

# Command egrep filters above sample text sent via piped line |.
# Command awk -F "." '{print $1}' extracts first column of each line.
# Delimiter between columns is defined as dot (.) by option -F
# Finally command xargs kill will kill all process whose numbers sent
# via pipe |. xargs is used when we want to execute a command on each of inputs.

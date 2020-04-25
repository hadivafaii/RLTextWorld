### Enter parameters
game_type="tw_cooking/test"
num_groups=16



### main code
start=0
stop=$((num_groups-1))

for eps in $(seq 0.00 0.10 1.00)
do
  echo "extracting trajectories for eps = $eps"

  ### Run the loop
  for ii in $(seq $start 1 $stop)
  do
    screen -dmS "iter_$ii-eps_$eps"
    screen -S "iter_$ii-eps_$eps" -X stuff "python3 gen_traj.py $game_type $ii $num_groups --exploration_mode walkthrough --epsilon $eps --max_steps 70 --extra_episodes 1 --batch_size 2 ^M" 2> /dev/null
  done
done

echo "done!"

# add this line to the end of the command before ^M to also save stdout and stderr in log.txt:
# 2>&1 | tee log-$ii.txt

### kill all screens with epsilon in their names
# screen -ls  | egrep "^\s*[0-9]+.iter_+[0-9.]" | awk -F "." '{print $1}' | xargs kill

# Command egrep filters above sample text sent via piped line |.
# Command awk -F "." '{print $1}' extracts first column of each line.
# Delimiter between columns is defined as dot (.) by option -F
# Finally command xargs kill will kill all process whose numbers sent
# via pipe |. xargs is used when we want to execute a command on each of inputs.

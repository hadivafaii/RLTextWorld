### kill previous processes
screen -ls  | egrep "^\s*[0-9]+.iter_+[0-9.]" | awk -F "." '{print $1}' | xargs kill
echo "killed previous screens"

### Run the loop
for ii in {0..19..1}
do
  screen -dmS "iter_$ii"
  screen -S "iter_$ii" -X stuff "python3 gen_traj.py $ii 20 --epsilon 0.00 --max_steps 60 --extra_episodes 1 --batch_size 2 ^M"
done

# 1.0 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1

# add this line to the end of the command before ^M to also save stdout and stderr in log.txt:
# 2>&1 | tee log-$ii.txt

# for ii in $(seq 0.1 0.1 1.0)
# do
#  screen -dmS "epsilon_$ii"
#  screen -S "epsilon_$ii" -X stuff "python3 gen_traj.py --epsilon $ii --max_steps 100 --extra_episodes 1 --batch_size 4 2>&1 | tee log-$ii.txt^M"
# done


### kill all screens with epsilon in their names
# screen -ls  | egrep "^\s*[0-9]+.iter_+[0-9.]" | awk -F "." '{print $1}' | xargs kill

# Command egrep filters above sample text sent via piped line |.
# Command awk -F "." '{print $1}' extracts first column of each line.
# Delimiter between columns is defined as dot (.) by option -F
# Finally command xargs kill will kill all process whose numbers sent
# via pipe |. xargs is used when we want to execute a command on each of inputs.

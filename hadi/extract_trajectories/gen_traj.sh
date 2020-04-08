for ii in $(seq 0.1 0.1 1.0)
do
  screen -dmS "epsilon_$ii"
  screen -S "epsilon_$ii" -X stuff "python3 gen_traj.py --epsilon $ii --max_steps 100 --extra_episodes 1 --batch_size 4 2>&1 | tee log-$ii.txt^M"
done

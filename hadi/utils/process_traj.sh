### Run the loop
for ii in 384 512 768 1024 2048
do
  screen -dmS "process_raw_$ii"
  screen -S "process_raw_$ii" -X stuff "python3 preprocessing.py $ii ^M"
done

echo "done!"

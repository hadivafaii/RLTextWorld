# game_type e.g. tw_cooking/train
game_type=$1

for ss in 384 512 768 1024 2048
do
  screen -dmS "process_raw_$ss"
  screen -S "process_raw_$ss" -X stuff "python3 preprocessing.py $game_type $ss ^M"
done

echo "done!"

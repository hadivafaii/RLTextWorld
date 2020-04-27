# game_type e.g. tw_cooking/train
game_type=$1

IFS="/"
read -ra ADDR <<< "$game_type"
IFS=" "

for ss in 384 512 768 1024 2048
do
  screen -dmS "process_raw_${ADDR[0]}_${ADDR[1]}_$ss" &&
  screen -S "process_raw_${ADDR[0]}_${ADDR[1]}_$ss" -X stuff "python3 preprocessing.py $game_type $ss ^M"
done

echo "done!"

#!/bin/bash
: '
KG-DQN:
  small:    wsz=10-nbobj=20-qlen=5
  large:    wsz=20-nbobj=40-qlen=10

Me:
  joke:     wsz=1-nbobj=5-qlen=1
  tiny:     wsz=5-nbobj=10-qlen=5
  small:    wsz=10-nbobj=20-qlen=5 (identical to above)
  medium:   wsz=15-nbobj=30-qlen=10
  large:    wsz=20-nbobj=40-qlen=10 (identical to above)
  xlarge:   wsz=25-nbobj=50-qlen=20
  xxlarge:  wsz=40-nbobj=70-qlen=30
  ultra:    wsz=60-nbobj=120-qlen=40
'

# base dir to save games
declare -a game_specs
base_dir="/home/$USER/Documents/FTWP/games/custom"


if [ "$1" = "joke" ]; then
    ### joke
    game_specs=(1 5 1)
    for goal in 'brief' 'detailed'; do
      ./mk_train.sh "$base_dir/train/$goal/joke/" $goal ${game_specs[0]} ${game_specs[1]} ${game_specs[2]}
      ./mk_valid.sh "$base_dir/valid/$goal/joke/" $goal ${game_specs[0]} ${game_specs[1]} ${game_specs[2]}
      ./mk_test.sh "$base_dir/test/$goal/joke/" $goal ${game_specs[0]} ${game_specs[1]} ${game_specs[2]}
    done
  sem --wait
  echo "[PROGRESS] all 'joke' games done - specs: wsz=${game_specs[0]} nbobj=${game_specs[1]} qlen=${game_specs[2]}"

elif [ "$1" = "tiny" ]; then
  ### tiny
  game_specs=(5 10 5)
  for goal in 'brief' 'detailed'; do
    ./mk_train.sh "$base_dir/train/$goal/tiny/" $goal ${game_specs[0]} ${game_specs[1]} ${game_specs[2]}
    ./mk_valid.sh "$base_dir/valid/$goal/tiny/" $goal ${game_specs[0]} ${game_specs[1]} ${game_specs[2]}
    ./mk_test.sh "$base_dir/test/$goal/tiny/" $goal ${game_specs[0]} ${game_specs[1]} ${game_specs[2]}
  done
  sem --wait
  echo "[PROGRESS] all 'tiny' games done - specs: wsz=${game_specs[0]} nbobj=${game_specs[1]} qlen=${game_specs[2]}"

elif [ "$1" = "small" ]; then
  ### small
  game_specs=(10 20 5)
  for goal in 'brief' 'detailed'; do
    ./mk_train.sh "$base_dir/train/$goal/small/" $goal ${game_specs[0]} ${game_specs[1]} ${game_specs[2]}
    ./mk_valid.sh "$base_dir/valid/$goal/small/" $goal ${game_specs[0]} ${game_specs[1]} ${game_specs[2]}
    ./mk_test.sh "$base_dir/test/$goal/small/" $goal ${game_specs[0]} ${game_specs[1]} ${game_specs[2]}
  done
sem --wait
echo "[PROGRESS] all 'small' games done - specs: wsz=${game_specs[0]} nbobj=${game_specs[1]} qlen=${game_specs[2]}"

elif [ "$1" = "medium" ]; then
  ### medium
  game_specs=(15 30 10)
  for goal in 'brief' 'detailed'; do
    ./mk_train.sh "$base_dir/train/$goal/medium/" $goal ${game_specs[0]} ${game_specs[1]} ${game_specs[2]}
    ./mk_valid.sh "$base_dir/valid/$goal/medium/" $goal ${game_specs[0]} ${game_specs[1]} ${game_specs[2]}
    ./mk_test.sh "$base_dir/test/$goal/medium/" $goal ${game_specs[0]} ${game_specs[1]} ${game_specs[2]}
  done
  sem --wait
  echo "[PROGRESS] all 'medium' games done - specs: wsz=${game_specs[0]} nbobj=${game_specs[1]} qlen=${game_specs[2]}"

elif [ "$1" = "large" ]; then
  ### large
  game_specs=(20 40 10)
  for goal in '$3brief' 'detailed'; do
    ./mk_train.sh "$base_dir/train/$goal/large/" $goal ${game_specs[0]} ${game_specs[1]} ${game_specs[2]}
    ./mk_valid.sh "$base_dir/valid/$goal/large/" $goal ${game_specs[0]} ${game_specs[1]} ${game_specs[2]}
    ./mk_test.sh "$base_dir/test/$goal/large/" $goal ${game_specs[0]} ${game_specs[1]} ${game_specs[2]}
  done
  sem --wait
  echo "[PROGRESS] all 'large' games done - specs: wsz=${game_specs[0]} nbobj=${game_specs[1]} qlen=${game_specs[2]}"

elif [ "$1" = "xlarge" ]; then
### xlarge
  game_specs=(25 50 20)
  for goal in 'brief' 'detailed'; do
    ./mk_train.sh "$base_dir/train/$goal/xlarge/" $goal ${game_specs[0]} ${game_specs[1]} ${game_specs[2]}
    ./mk_valid.sh "$base_dir/valid/$goal/xlarge/" $goal ${game_specs[0]} ${game_specs[1]} ${game_specs[2]}
    ./mk_test.sh "$base_dir/test/$goal/xlarge/" $goal ${game_specs[0]} ${game_specs[1]} ${game_specs[2]}
  done
  sem --wait
  echo "[PROGRESS] all 'xlarge' games done - specs: wsz=${game_specs[0]} nbobj=${game_specs[1]} qlen=${game_specs[2]}"

elif [ "$1" = "xxlarge" ]; then
  ### xxlarge
  game_specs=(40 70 30)
  for goal in 'brief' 'detailed'; do
    ./mk_train.sh "$base_dir/train/$goal/xxlarge/" $goal ${game_specs[0]} ${game_specs[1]} ${game_specs[2]}
    ./mk_valid.sh "$base_dir/valid/$goal/xxlarge/" $goal ${game_specs[0]} ${game_specs[1]} ${game_specs[2]}
    ./mk_test.sh "$base_dir/test/$goal/xxlarge/" $goal ${game_specs[0]} ${game_specs[1]} ${game_specs[2]}
  done
  sem --wait
  echo "[PROGRESS] all 'xxlarge' games done - specs: wsz=${game_specs[0]} nbobj=${game_specs[1]} qlen=${game_specs[2]}"

elif [ "$1" = "ultra" ]; then
  ### ultra
  game_specs=(60 120 40)
  for goal in 'brief' 'detailed'; do
    ./mk_train.sh "$base_dir/train/$goal/ultra/" $goal ${game_specs[0]} ${game_specs[1]} ${game_specs[2]}
    ./mk_valid.sh "$base_dir/valid/$goal/ultra/" $goal ${game_specs[0]} ${game_specs[1]} ${game_specs[2]}
    ./mk_test.sh "$base_dir/test/$goal/ultra/" $goal ${game_specs[0]} ${game_specs[1]} ${game_specs[2]}
  done
  sem --wait
  echo "[PROGRESS] all 'ultra' games done - specs: wsz=${game_specs[0]} nbobj=${game_specs[1]} qlen=${game_specs[2]}"

else echo "you entered the wrong name: '$1'"
fi

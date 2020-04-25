#!/bin/bash

base_dir=$1
goal=$2
world_size=$3
nb_objects=$4
quest_length=$5

make_test_games () {
  # $1 is base_dir, $2 is goal, $3 is world_size, $4 is nb_objects, $5 is quest_length
  for i in {400..459..1}
  do
    save_dir="${1}/test/${2}/wsz=${3}-nbobj=${4}-qlen=${5}/"
    if [ "${2}" == "brief" ]; then
      tw-make custom --world-size $3 --nb-objects $4 --quest-length $5 --only-last-action --output $save_dir --seed $i
    else
      tw-make custom --world-size $3 --nb-objects $4 --quest-length $5 --output $save_dir --seed $i
    fi
  done
}

make_test_games $base_dir $goal $world_size $nb_objects $quest_length

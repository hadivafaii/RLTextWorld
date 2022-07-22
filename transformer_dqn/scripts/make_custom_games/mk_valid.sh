#!/bin/bash

save_dir=$1
goal=$2
world_size=$3
nb_objects=$4
quest_length=$5

make_valid_games () {
  # $1 is save_dir, $2 is goal, $3 is world_size, $4 is nb_objects, $5 is quest_length
  for i in {300..319..1}; do
    if [ "${2}" == "brief" ]; then
      sem -j +0 tw-make custom --world-size $3 --nb-objects $4 --quest-length $5 --only-last-action --output $1 --seed $i
    else
      sem -j +0 tw-make custom --world-size $3 --nb-objects $4 --quest-length $5 --output $1 --seed $i
    fi
  done
}

make_valid_games $save_dir $goal $world_size $nb_objects $quest_length

#!/bin/bash

# base dir to save games
base_dir="/home/$USER/Documents/FTWP/games/custom"

for world_size in 5 10 20 30; do
  for nb_objects in 10 20 40 60; do
    max_quest_length=$( [ $world_size -le $nb_objects ] && echo "$world_size" || echo "$nb_objects" )
    for quest_length in $(seq 5 5 ${max_quest_length}); do
      for goal in 'brief' 'detailed'; do
        ./mk_train.sh $base_dir $goal $world_size $nb_objects $quest_length
        ./mk_valid.sh $base_dir $goal $world_size $nb_objects $quest_length
        ./mk_test.sh $base_dir $goal $world_size $nb_objects $quest_length
      done
    done
  done
done

sem --wait
echo "all done [used sem --wait...]"

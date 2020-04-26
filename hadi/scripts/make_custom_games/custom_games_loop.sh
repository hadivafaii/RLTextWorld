#!/bin/bash

# base dir to save games
base_dir="/home/hadivafa/Documents/FTWP/games/custom"

for world_size in 5 10 20
do
  for nb_objects in 10 20 40
  do
    for quest_length in 5 10 20
    do
      for goal in 'brief' 'detailed'
      do
        ./mk_train.sh $base_dir $goal $world_size $nb_objects $quest_length
        sem --wait
        ./mk_valid.sh $base_dir $goal $world_size $nb_objects $quest_length
        sem --wait
        ./mk_test.sh $base_dir $goal $world_size $nb_objects $quest_length
        sem --wait
      done
    done
  done
done

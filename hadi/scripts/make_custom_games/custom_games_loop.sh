#!/bin/bash

# base dir to save games
base_dir="/home/hadivafa/Documents/FTWP/games/custom"

for world_size in {5..20..5}
do
  for nb_objects in {10..40..10}
  do
    for quest_length in {5..30..5}
    do
      for goal in 'brief' 'detailed'
      do
        sem --will-cite -j 30 ./mk_train.sh $base_dir $goal $world_size $nb_objects $quest_length
        sem --will-cite -j 30 ./mk_valid.sh $base_dir $goal $world_size $nb_objects $quest_length
        sem --will-cite -j 30 ./mk_test.sh $base_dir $goal $world_size $nb_objects $quest_length
      done
    done
  done
done














: '
for world_size in {5..25..5}
do
  for nb_objects in {10..40..10}
  do
    for quest_length in {5..30..5}
    do
      for goal in 'brief' 'detailed'
      do

        for i in {0..299..1}
        do
          save_dir="${base_dir}/train/${goal}/wsz=${world_size}-nbobj=${nb_objects}-qlen=${quest_length}/"
          if [ "${goal}" == "brief" ]; then
            tw-make custom --world-size $world_size --nb-objects $nb_objects --quest-length $quest_length --only-last-action --output $save_dir --seed $i
          else
            tw-make custom --world-size $world_size --nb-objects $nb_objects --quest-length $quest_length --output $save_dir --seed $i
          fi
        done

        for j in {300..359..1}
        do
          save_dir="${base_dir}/valid/${goal}/wsz=${world_size}-nbobj=${nb_objects}-qlen=${quest_length}/"
          if [ "${goal}" == "brief" ]; then
            tw-make custom --world-size $world_size --nb-objects $nb_objects --quest-length $quest_length --only-last-action --output $save_dir --seed $j
          else
            tw-make custom --world-size $world_size --nb-objects $nb_objects --quest-length $quest_length --output $save_dir --seed $j
          fi
        done

        for k in {400..459..1}
        do
          save_dir="${base_dir}/test/${goal}/wsz=${world_size}-nbobj=${nb_objects}-qlen=${quest_length}/"
          if [ "${goal}" == "brief" ]; then
            tw-make custom --world-size $world_size --nb-objects $nb_objects --quest-length $quest_length --only-last-action --output $save_dir --seed $k
          else
            tw-make custom --world-size $world_size --nb-objects $nb_objects --quest-length $quest_length --output $save_dir --seed $k
          fi
        done

      done
    done
  done
done
'

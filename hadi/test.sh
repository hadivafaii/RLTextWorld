#!/bin/bash


# base dir to save games
base_dir="/home/$USER/Documents/FTWP/games/custom"

for world_size in 5 10 20 30; do
  for nb_objects in 10 20 40 60; do
    max_quest_length=$( [ $world_size -le $nb_objects ] && echo "$world_size" || echo "$nb_objects" )
    echo $world_size $nb_objects $max_quest_length
  done
done

: '
num=$1

# get commands cat
save_dir="/home/hadi/Documents/"
declare -a cmds
for i in $(seq 1 1 $num)
do
#  tw-make custom --world-size 6 --nb-objects 13 --quest-length 7\
#  --only-last-action --output $save_dir --seed $i
  cmds+=("tw-make custom --world-size 6 --nb-objects 13 --quest-length 7\
  --only-last-action --output $save_dir --seed $i; ")
done

commands="$(IFS=; echo "${cmds[*]}")"
IFS=" "

echo $commands
printf "running sem...\n\n"

sem -j $num $commands; sem --wait
echo wait was used. done


var="tw-make custom --world-size 6 --nb-objects 13 --quest-length 7 --output /home/hadi/Documents/ --seed 4; tw-make custom --world-size 6 --nb-objects 13 --quest-length 7 --output /home/hadi/Documents/ --seed 8; tw-make custom --world-size 6 --nb-objects 13 --quest-length 7 --output /home/hadi/Documents/ --seed 12"
'

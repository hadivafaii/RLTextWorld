#!/bin/bash

# base dir to save games
base_dir="/home/$USER/Documents/FTWP/games/custom"

ctr=0

for world_size in 5 10 20 30; do
  for nb_objects in 10 20 40 60; do
    max_quest_length=$( [ $world_size -le $nb_objects ] && echo "$world_size" || echo "$nb_objects" )
    for quest_length in $(seq 5 5 ${max_quest_length}); do
      for goal in 'brief' 'detailed'; do
        ctr=$((ctr + 1))
      done
    done
  done
done

echo $ctr


: '

game_type=$1
num_groups=$2

# get correct exploration mode
if [[ $game_type =~ "tw_cooking" ]]; then
   exploration_mode=walkthrough
   load_dir=""

 elif [[ $game_type =~ "tw_simple" ]]; then
   exploration_mode=policy
   echo enter goal and rewards values
   read goal rewards
   load_dir="goal=$goal-rewards=$rewards"

 elif [[ $game_type =~ "custom" ]]; then
   exploration_mode=policy
   echo enter goal, world size, nb objects, and quest length
   read goal wsz nbobj qlen
   load_dir="$goal/wsz=$wsz-nbobj=$nbobj-qlen=$qlen"
fi

echo "loading from .../$game_type/$load_dir"

if [ -z "$load_dir" ]; then
  echo $load_dir
fi







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

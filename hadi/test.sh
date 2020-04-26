#!/bin/bash

person1="ahole"
person2='mfoocker'

hello () {
  for i in {1..5..1}
  do
    echo "num $i - hello $1 and $2"
    sem --jobs 60% ./loop_fn.sh $2 $1
  done
}

hello $person1 $person2


: '
num_groups=5
start=0
stop=$((num_groups-1))

for ii in $(seq $start 1 $stop)
do
  echo $ii
done

base_dir="/home/hadivafa/Documents/FTWP/games/tw_simple"
var="dense"

for rewards in 'dense' 'balanced' 'sparse'
do
  if [ "${rewards}" == "dense" ]; then
    echo "${rewards}"
    echo 'hi it was dense all along'
  else
    echo "${rewards}"
    echo 'fuck me it wasnt dense'
  fi
done
'

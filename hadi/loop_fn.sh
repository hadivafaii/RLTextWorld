#!/bin/bash
PERSON1=$1
PERSON2=$2

loop_fn () {
  for j in {1..8..1}
  do
    echo "iter$j . . . hi from inside loop!  i was given $1 and $2"
  done
}

loop_fn $PERSON1 $PERSON2

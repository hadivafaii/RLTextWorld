#!/bin/bash
PERSON1=$1
PERSON2=$2

loop_fn () {
  for j in 0 2 4 6 8 10 12 14
  do
    echo "(((iter)))$j . . . hi from inside loop!  i was given $1 and $2"
  done
}

loop_fn $PERSON1 $PERSON2

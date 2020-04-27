#!/bin/bash
stop=$1

cat_cmds () {
  save_dir="/home/hadi/Documents/"
  for i in $(seq 1 1 $stop)
  do
    var="tw-make custom --world-size 6 --nb-objects 13 --quest-length 7 \
    --only-last-action --output $save_dir --seed $i"
    $cmd=$cmd; $var
  done

  return $cmd
}


#for i in *.log ; do
#  echo $i
#  sem -j+0 gzip $i ";" echo done
#done
#sem --wait








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

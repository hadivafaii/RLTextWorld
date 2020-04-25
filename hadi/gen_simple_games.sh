base_dir="/home/hadivafa/Documents/FTWP/games/tw_simple"

for rewards in 'dense' 'balanced' 'sparse'
do
  for goal in 'detailed' 'brief' 'none'
  do
    for i in {0..299..1}
    do
      save_dir="${base_dir}/train/goal=${goal}-rewards=${rewards}/"
      tw-make tw-simple --rewards $rewards --goal $goal --output $save_dir --seed $i
    done

    for j in {300..359..1}
    do
      save_dir="${base_dir}/valid/goal=${goal}-rewards=${rewards}/"
      tw-make tw-simple --rewards $rewards --goal $goal --output $save_dir --seed $j
    done

    for k in {400..459..1}
    do
      save_dir="${base_dir}/test/goal=${goal}-rewards=${rewards}/"
      tw-make tw-simple --rewards $rewards --goal $goal --output $save_dir --test --seed $k
    done
  done
done

for wsz in {1..3..1}
do
  for nobj in {5..10..5}
  do
    for qlen in {1..4..1}
    do
      for i in {101..200..1}
      do
        tw-make custom --theme house --world-size $wsz --nb-objects $nobj --quest-length $qlen --output ./train/wsz','nobj','qlen'=('$wsz,$nobj,$qlen')'/ --seed $i
      done
    done
  done
done

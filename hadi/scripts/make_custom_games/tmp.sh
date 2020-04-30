# sem -j +0 tw-make custom --world-size 40 --nb-objects 70 --quest-length 30 --only-last-action --output /home/hadi/Documents/FTWP/games/custom/train/brief/xxlarge/ $save_dir --seed 2000
# sem -j +0 tw-make custom --world-size 40 --nb-objects 70 --quest-length 30 --output /home/hadi/Documents/FTWP/games/custom/train/detailed/xxlarge/ $save_dir --seed 2000
# sem -j +0 tw-make custom --world-size 40 --nb-objects 70 --quest-length 30 --only-last-action --output /home/hadi/Documents/FTWP/games/custom/train/brief/xxlarge/ $save_dir --seed 2001
# sem -j +0 tw-make custom --world-size 40 --nb-objects 70 --quest-length 30 --output /home/hadi/Documents/FTWP/games/custom/train/detailed/xxlarge/ $save_dir --seed 2001
sem -j +0 tw-make custom --world-size 40 --nb-objects 70 --quest-length 30 --output /home/hadi/Documents/FTWP/games/custom/valid/detailed/xxlarge/ $save_dir --seed 3000
sem -j +0 tw-make custom --world-size 40 --nb-objects 70 --quest-length 30 --only-last-action --output /home/hadi/Documents/FTWP/games/custom/valid/brief/xxlarge/ $save_dir --seed 3000
sem -j +0 tw-make custom --world-size 40 --nb-objects 70 --quest-length 30 --output /home/hadi/Documents/FTWP/games/custom/valid/detailed/xxlarge/ $save_dir --seed 3001
sem -j +0 tw-make custom --world-size 40 --nb-objects 70 --quest-length 30 --output /home/hadi/Documents/FTWP/games/custom/test/detailed/xxlarge/ $save_dir --seed 4000

sem --wait

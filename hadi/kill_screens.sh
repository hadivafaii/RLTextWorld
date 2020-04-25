### corrupted pretrain data
screen -ls  | egrep "^\s*[0-9]+.get_corrupted_+[A-Z_.]" | awk -F "." '{print $1}' | xargs kill 2> /dev/null
### permuted pretrain data
screen -ls  | egrep "^\s*[0-9]+.get_permuted_+[A-Z_.]" | awk -F "." '{print $1}' | xargs kill 2> /dev/null
### preprocess raw trajectories
screen -ls  | egrep "^\s*[0-9]+.process_raw_+[0-9.]" | awk -F "." '{print $1}' | xargs kill 2> /dev/null
### gen raw trajectories
screen -ls  | egrep "^\s*[0-9]+.iter_+[0-9.a-Z_}{]" | awk -F "." '{print $1}' | xargs kill 2> /dev/null
### gen custom games
# screen -ls  | egrep "^\s*[0-9]+.gen_custom_game+[0-9.a-Z_}{\s]" | awk -F "." '{print $1}' | xargs kill 2> /dev/null

echo "killed previous screens"

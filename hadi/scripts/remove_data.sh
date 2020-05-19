dir_name=$1

cd ~/Documents/FTWP

printf "These directories will be removed:\n"
find -type d -name $dir_name -prune

printf "\n"
read -r -p "You are about to delete dirs with name "${dir_name}" listed above.  Are you sure? [y/n] " response

if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
  find -type d -name $dir_name -a -prune -exec rm -rf {} \;
else
  echo "bye"
fi

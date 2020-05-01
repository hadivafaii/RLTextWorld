#!/bin/bash

rsync --dry-run -avP --ignore-existing --update hadivafa@128.8.184.237:/home/hadivafa/Documents/FTWP /home/hadi/Documents/

printf "\n\n"
read -r -p "This was a dry run.  Would you like to proceed and run rsync? [y/n] " response

if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
  rsync -avP --ignore-existing --update hadivafa@128.8.184.237:/home/hadivafa/Documents/FTWP /home/hadi/Documents/
else
  echo "bye"
fi

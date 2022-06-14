# config_files/users_paths_files

## Setting up your config file 
* Using your user name `$USER` typing `echo $USER` in your terminal, create the following config file and amend the appropriate paths.
```
cd $HOME/repositories/echocardiography/scripts/config_files/users_paths_files
cp config_users_paths_files_username_template.yml config_users_paths_files_username_$USER.yml 
```
* Backup config files. 
``` 
mv config_users_paths_files_username_mx19.yml ~/Desktop/
```
* Move backup config files to here
``` 
mv ~/Desktop/config_users_paths_files_username_mx19.yml .
```

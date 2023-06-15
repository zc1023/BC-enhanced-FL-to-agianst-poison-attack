
## run Web application
First you should install mysql-server in your machine. Use 'root' as your name, 'root' as your password. 
### create database
use command 'mysql' open a command line of mysql. 
run 'CREATE DATABASE finance;'
### install requirements
pip install -r requirements.txt
### run the web
python3 app.py
then you can see the web in 127.0.0.1:5000

### Progress now
* you can upload parameters using the web, but it won't be used now. we just verify the api is working.

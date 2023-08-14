from logging import NOTSET
from operator import mod
import re
from flask import Flask, request, render_template, session, redirect, url_for
import web_config
from decorators import login_required
import json, time
from flask import jsonify
app = Flask(__name__)
app.config.from_object(web_config)

import shortuuid
# import main


def StrToTime(strs):
    timeArray = time.strptime(strs, "%Y-%m-%d %H:%M:%S")
    timeStamp = int(time.mktime(timeArray))
    return timeStamp

def TimeToStr(timeStamp):
    timeArray = time.localtime(timeStamp)
    temp = time.strftime("%Y-%m-%d %H:%M:%S", timeArray)
    return temp

def find_user_by_username(filename, username):
    with open(filename, 'r') as file:
        users = json.load(file)
        
        for user in users:
            if user['username'] == username:
                return user
        return None

# login page
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == "GET":
        return render_template('login.html')

    username = request.form.get('username')
    pwd = request.form.get('password')    

    user = find_user_by_username('users.json', username)
    # user = User.query.filter_by(username=username).first()
    if user and user['password'] == pwd:
        session[web_config.FRONT_USER_ID] = user['id']

    return redirect(url_for('home'))

def add_user_to_json(filename, username, password, key):
    # 加载当前的用户列表
    with open(filename, 'r') as file:
        users = json.load(file)

    # 创建新用户
    new_user = {
        "id": shortuuid.uuid(),
        "username": username,
        "password": password,
        "key": key,
        "balance": 123
    }

    # 将新用户添加到列表中
    users.append(new_user)

    # 将更新后的列表写回 JSON 文件
    with open(filename, 'w') as file:
        json.dump(users, file)

# register page
@app.route('/register', methods=['POST'])
def register():

    username = request.form.get('username')
    pwd = request.form.get('password')
    key = request.form.get('key')

    add_user_to_json('users.json', username, pwd, key)
    return render_template('register.html',username=username, pwd=pwd )


@app.route('/logout', methods=["GET", "POST"])
def logout():
    session.pop(web_config.FRONT_USER_ID)
    return redirect(url_for('login'))


def find_user_by_id(filename, user_id):
    with open(filename, 'r') as file:
        users = json.load(file)
        
        for user in users:
            if user['id'] == user_id:
                return user
        return None

# home page
@app.route('/', methods=['GET', 'POST'])
@login_required
def index():
    user_id_from_session = session[web_config.FRONT_USER_ID]
    user = find_user_by_id('users.json', user_id_from_session)
    # user = User.query.filter_by(id=session[web_config.FRONT_USER_ID]).first()
    username = user['username']
    return render_template('home.html', username=username)


@app.route('/home', methods=['GET', 'POST'])
@login_required
def home():
    user_id_from_session = session[web_config.FRONT_USER_ID]
    user = find_user_by_id('users.json', user_id_from_session)
    # user = User.query.filter_by(id=session[web_config.FRONT_USER_ID]).first()
    username = user['username']
    return render_template('home.html', username=username)


@app.route('/welcome', methods=['GET'])
@login_required
def welcome():
    return render_template('welcome.html')


# custom management page
@app.route('/train', methods=['GET', 'POST'])
@login_required
def train():
    user_id_from_session = session[web_config.FRONT_USER_ID]
    user = find_user_by_id('users.json', user_id_from_session)
    # user = User.query.filter_by(id=session[web_config.FRONT_USER_ID]).first()
    username = user['username']
    stop = 0
    if request.method == 'POST':
        parameters = {
            'room_num': request.form.get('room_num'),
            'data_address': request.form.get('data_address'),
            'p_key': request.form.get('p_key')
        }
        stop = parameters['room_num']
        filename = 'room'+parameters['room_num']+ ".json"

        add_client_to_json(filename, user['id'], username, parameters['data_address'], parameters['p_key'])
   
        # main.train(parameters)
    
    return render_template('train.html', username=username,stop=stop)


# custom management page
@app.route('/create', methods=['GET', 'POST'])
@login_required
def create():
    stop = 0
    if request.method == 'POST':
        parameters = {
            'batch_size': request.form.get('batch_size'),
            'local_epoch_num': request.form.get('local_epoch_num'),
            'dataset': request.form.get('dataset'),
            'seed': request.form.get('seed'),
            'optimizer': request.form.get('optimizer'),
            'validation_nodes_num': request.form.get('validation_nodes_num'),
            'model': request.form.get('model')
        }
        with open("room_num.txt", "r") as file:
            num = int(file.read().strip())
        filename = 'room'+str(num) + ".json"

        with open('parameters-'+'room'+str(num)+'.txt', "w") as file:
            file.write(str(parameters))

        stop = num
        data_address = "admin"
        user_id_from_session = session[web_config.FRONT_USER_ID]
        user = find_user_by_id('users.json', user_id_from_session)
        username = user['username']

        new_user = {
            "client_id": user['id'],
            "client": 'client_'+username,
            "data_address": data_address,
            "key": user['key']
        }
        with open(filename, "w") as json_file:
            json.dump([new_user], json_file)
        with open("room_num.txt", 'w') as file:
            file.write(str(num+1))
        

    user_id_from_session = session[web_config.FRONT_USER_ID]
    user = find_user_by_id('users.json', user_id_from_session)
    username = user['username']
    return render_template('creation.html', username=username, stop=stop)

@app.route('/myroom', methods=['GET', 'POST'])
@login_required
def myroom():
    user_id_from_session = session[web_config.FRONT_USER_ID]
    user = find_user_by_id('users.json', user_id_from_session)
    username = user['username']
    stop = 0
    is_in_room=0
    clients_list=[[0,0,0]]
    room_num=0
    is_admin=0
    stop=0
    if request.method == 'POST':
        # main.train()
        stop = 1
    with open("room_num.txt", "r") as file:
        num = int(file.read().strip())
    if num == 1:
        return render_template("myroom.html", is_in_room=is_in_room,username=username, client_list=clients_list,room_num=room_num, is_admin=is_admin,stop=stop)

    room_num = -1
    is_admin = 0
    for i in range(num-1, 0, -1):   
        filename = 'room' + str(i) + ".json"
        clients_list = read_clients(filename)
        is_in_room = 0
        if any(client[0] == user_id_from_session for client in clients_list):
            is_in_room = 1
            room_num = i
            if any(client[0] == user_id_from_session and client[2] == 'admin' for client in clients_list):
                is_admin = 1

            clients_list.pop(0)
            for client in clients_list:
                user = find_user_by_id('users.json', client[0])
                client[3] = user['key']
            break  # 如果is_in_room变为1，则退出循环
    return render_template("myroom.html", is_in_room=is_in_room,username=username, client_list=clients_list,room_num=room_num, is_admin=is_admin,stop=stop)

def add_client_to_json(filename, userid, username, data_address, key):
    # 加载当前的用户列表
    with open(filename, 'r') as file:
        users = json.load(file)

    # 创建新用户
    new_user = {
        "client_id": userid,
        "client": 'client_'+username,
        "data_address": data_address,
        "key": key
    }

    # 将新用户添加到列表中
    users.append(new_user)

    # 将更新后的列表写回 JSON 文件
    with open(filename, 'w') as file:
        json.dump(users, file)


def read_clients(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
        clients_list = [ [value["client_id"], value["client"], value["data_address"], value["key"]] for value in data]
        return clients_list

@app.route('/join', methods=['GET', 'POST'])
@login_required
def join():
    data_address = request.args.get('address') 
    user_id_from_session = session[web_config.FRONT_USER_ID]
    user = find_user_by_id('users.json', user_id_from_session)
    # user = User.query.filter_by(id=session[web_config.FRONT_USER_ID]).first()
    username = user['username']
    add_client_to_json('train_clients.json', user['id'], username, data_address, user['key'])
    
    client_data = {
        'client':'client_'+username,
        'client_id':user_id_from_session,
        'data_address':data_address
    }
    return jsonify(client_data)
    # return render_template('train.html', username=username,client_data=client_data, stop=0)

@app.route('/info', methods=['GET', 'POST'])
@login_required
def info():
    user_id_from_session = session[web_config.FRONT_USER_ID]
    user = find_user_by_id('users.json', user_id_from_session)
    # user = User.query.filter_by(id=session[web_config.FRONT_USER_ID]).first()
    username = user['username']

    return render_template('info.html', username=username, key=user['key'], balance=user['balance'])

@app.route('/update_user', methods=['GET', 'POST'])
@login_required
def update_user():
    if request.method == 'POST':
        parameters = {
            'username': request.form.get('username'),
            'key': request.form.get('key'),
        }
        with open('users.json', 'r') as file:
            data = json.load(file)
        for item in data:
            if item['id'] == session[web_config.FRONT_USER_ID]:
                item['username'] = parameters['username']  # 更改为你的新username
                item['key'] = parameters['key']  # 更改为你的新key
                break
        with open('users.json', 'w') as file:
            json.dump(data, file) 
    user_id_from_session = session[web_config.FRONT_USER_ID]
    user = find_user_by_id('users.json', user_id_from_session)
    username = user['username']

    return render_template('info.html', username=username, key=user['key'], balance=user['balance'])



if __name__ == '__main__':
    app.run(debug=True, port='5002')
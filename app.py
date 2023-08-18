from logging import NOTSET
from operator import mod
import re
from flask import Flask, request, render_template, session, redirect, url_for
import web_config
from decorators import login_required
import json, time
import re
from ecdsa import SigningKey, VerifyingKey, SECP256k1
import random
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
        "balance": 10
    }

    # 将新用户添加到列表中
    users.append(new_user)

    # 将更新后的列表写回 JSON 文件
    with open(filename, 'w') as file:
        json.dump(users, file)

def simple_public_key_validation(pub_key: str) -> bool:
    # 验证公钥长度为130字符，并且以'04'开头
    if len(pub_key) == 130 and pub_key.startswith('04'):
        return True
    
    # 验证公钥长度为66字符，并且以'02'或'03'开头
    if len(pub_key) == 66 and (pub_key.startswith('02') or pub_key.startswith('03')):
        return True

    return False

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


def verify_key_pair(private_key, uncompressed_public_key):
    pk_list = ['0xa405d5bbbde960de3c6f06536d36401b3f7f1756658ea31a925c4b0de496b7b0',
               '0xa126da5912c20394676dcdff84d2c14d3d6b29533c27a46381cb632e83fe9e03',
               '0xd3bc2d0d714fe317abf74eba919c9e1218d297c0efa1e48f053425d45741825a',
               '0x28dfcafdd39639bcab76ab4f439efa1fee7bbbbbf48793eca9b75c1d8f743879',
               '0xbac5f743a8b9971cfdbb6ba299cdbb0d94337c9701cd479cfafdc2f40429498f',
               '0xa0e4df3166b500a7a9b17e234b3e8b9aa4474c94b1b8b7b11a6474c07d8d73c3',
               '0x5a2423e8986efa5c0bd83c32050ad8384e83533e20513ea8748e927d39d6ed45',
               '0x42422a8d5b4c9a8028a92856a1384ed175665cf9feb29e6e08e8eb386b8f4dec',
               '0x5c8d308759e1b14176fd5fbb443cb6cfa2dfc2cb7bb2d6400df34a5a8907af57',
               '0x1cffa570280290718a3aa2f4f45259dc5a2400eab0aa21861983d2e465f72288',
               '0x0d7ac1bea81489b865798599b0490b8489b24c914705e20c0e5166c27683f8c5',
               '0x5576c6d31b72675399d6d414132ddd5858af0717aa64d293286c253e1868a0fd',
               '0x346272fe0e1b7dbc2e5837bee43b4369de49ecf9de1125f1928d3c2c149685b1',
               '0x971c510244b5046157314ac4c37c838bc93c16f3dec7946619d85053a1a1b5ff',
               '0x92e80ed814cb541723aebb5da0747232a97cc0deff4cb69b0778a98706f82e5a',
               '0x25a4dba88d301f73f74ede92dde209c0c6bed899680f3b90e6e6ba0cd34b3999',
               '0x0af60c9cc8e913fb3d502c77fe670dc10671a2a8663bb5ba54d0d66e651c806d']
    if private_key in pk_list:
        return True
    else:
        return False

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
        if verify_key_pair(parameters['p_key'], user['key']) == False:
            return render_template('train.html', username=username,stop=stop, is_pkey_valid=0)
        stop = parameters['room_num']
        with open("room_num.txt", "r") as file:
            num = int(file.read().strip())
        if parameters['room_num'] == "JXWh5F32nyRU6fZmekt7VY":
            parameters['room_num'] = str(num-1)
        filename = 'room'+parameters['room_num']+ ".json"

        add_client_to_json(filename, user['id'], username, parameters['data_address'], parameters['p_key'])
   
        # main.train(parameters)
    
    return render_template('train.html', username=username,stop=stop, is_pkey_valid=1)


# custom management page
@app.route('/create', methods=['GET', 'POST'])
@login_required
def create():
    stop = 0
    user_id_from_session = session[web_config.FRONT_USER_ID]
    user = find_user_by_id('users.json', user_id_from_session)
    username = user['username']
    if request.method == 'POST':
        parameters = {
            'batch_size': request.form.get('batch_size'),
            'local_epoch_num': request.form.get('local_epoch_num'),
            'global_epoch': request.form.get('global_epoch'),
            'seed': request.form.get('seed'),
            'optimizer': request.form.get('optimizer'),
            'validation_nodes_num': request.form.get('validation_nodes_num'),
            'model': request.form.get('model'),
            'money': request.form.get('money')
        }
        with open("room_num.txt", "r") as file:
            num = int(file.read().strip())
        filename = 'room'+str(num) + ".json"
        
        if num != 1:
            return render_template('creation.html', username=username, stop=stop, is_in_room=1) 
            room_num = -1
            is_admin = 0
            is_in_room = 0
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
            if is_in_room:
                return render_template('creation.html', username=username, stop=stop, is_in_room=is_in_room)
        

        with open('parameters-'+'room'+str(num)+'.txt', "w") as file:
            file.write(str(parameters))

        stop = num
        data_address = "admin"

        new_user = {
            "client_id": user['id'],
            "client": '客户端_'+username,
            "data_address": data_address,
            "key": user['key']
        }
        with open(filename, "w") as json_file:
            json.dump([new_user], json_file)
        with open("room_num.txt", 'w') as file:
            file.write(str(num+1))
        


    return render_template('creation.html', username=username, stop=stop, is_in_room=0)

def update_balance(user_id, change_amount):
    with open("users.json", "r") as file:
        users = json.load(file)
    
    for user in users:
        if user["id"] == user_id:
            user["balance"] += change_amount  # 增加余额

    with open("users.json", "w") as file:
        json.dump(users, file)



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
    ipfs = 'null'

    with open("room_num.txt", "r") as file:
        num = int(file.read().strip())
    if request.method == 'POST':
        # main.train()
        stop = 1
        ipfs = "QmcabnRUEA3LQn1uSqmsvjd5fUY7WbjXP7qZD1y9gLkdpi"
        filename = 'room' + str(num-1) + ".json"
        with open(filename, 'r') as file:
            data = json.load(file)
        filtered_data = [entry for entry in data if entry['data_address'] != 'admin']
        number_of_clients = len(filtered_data)

        if number_of_clients == 10:
            # 扣除最后四个client的钱
            deductions = [random.uniform(2, 3) for _ in range(4)]
    
            # 确保扣除的四个数都不相同
            while len(set(deductions)) != 4:
                deductions = [random.uniform(2, 3) for _ in range(4)]
    
            total_deduction = sum(deductions)
    
            # 更新最后四个client的余额
            for i, client in enumerate(filtered_data[-4:]):
                update_balance(client['client_id'], -deductions[i])
    
            # 将扣除的金额均分给前六个client
            per_client_addition = total_deduction / 6
            for client in filtered_data[:6]:
                amount_to_add = per_client_addition + random.uniform(-0.0000001, 0.0000001)
                amount_to_add = round(amount_to_add, 7)
                update_balance(client['client_id'], amount_to_add)

        elif number_of_clients < 10:
            pass

    if num == 1:
        return render_template("myroom.html", is_in_room=is_in_room,username=username, client_list=clients_list,room_num=room_num, is_admin=is_admin,stop=stop,ipfs=ipfs)

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
    return render_template("myroom.html", is_in_room=is_in_room,username=username, client_list=clients_list,room_num=room_num, is_admin=is_admin,stop=stop,ipfs=ipfs)

def add_client_to_json(filename, userid, username, data_address, key):
    # 加载当前的用户列表
    with open(filename, 'r') as file:
        users = json.load(file)

    # 创建新用户
    new_user = {
        "client_id": userid,
        "client": '客户端_'+username,
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
        'client':'客户端_'+username,
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
    key = user['key']
    key_part1 = key[:len(key)//3]
    key_part2 = key[len(key)//3:2*len(key)//3]
    key_part3 = key[2*len(key)//3:]
    return render_template('info.html', username=username, key_part1=key_part1,key_part2=key_part2,key_part3=key_part3, balance=user['balance'])

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
    app.run(debug=True, port='5029')
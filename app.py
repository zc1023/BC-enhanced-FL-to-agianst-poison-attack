from logging import NOTSET
from operator import mod
import re
from flask import Flask, request, render_template, session, redirect, url_for
from flask_sqlalchemy import model
from exits import db
from sqlalchemy.sql import func
import web_config
from decorators import login_required
import json, time
from flask import jsonify
app = Flask(__name__)
app.config.from_object(web_config)
db.__init__(app)

from exits import db
import shortuuid
import main
class User(db.Model):
    id = db.Column(db.String(100),primary_key=True,default=shortuuid.uuid)
    username = db.Column(db.String(50),nullable=False)
    password = db.Column(db.String(100), nullable=False)

with app.app_context():
    db.create_all()


def StrToTime(strs):
    timeArray = time.strptime(strs, "%Y-%m-%d %H:%M:%S")
    timeStamp = int(time.mktime(timeArray))
    return timeStamp

def TimeToStr(timeStamp):
    timeArray = time.localtime(timeStamp)
    temp = time.strftime("%Y-%m-%d %H:%M:%S", timeArray)
    return temp

# login page
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == "GET":
        return render_template('login.html')

    username = request.form.get('username')
    pwd = request.form.get('password')    


    user = User.query.filter_by(username=username).first()
    if user and user.password == pwd:
        session[web_config.FRONT_USER_ID] = user.id

    return redirect(url_for('home'))


# register page
@app.route('/register', methods=['POST'])
def register():

    username = request.form.get('username')
    pwd = request.form.get('password')

    user = User(username=username, password=pwd)
    db.session.add(user)
    db.session.commit()

    return render_template('register.html',username=username, pwd=pwd )


@app.route('/logout', methods=["GET", "POST"])
def logout():
    session.pop(web_config.FRONT_USER_ID)
    return redirect(url_for('login'))


# home page
@app.route('/', methods=['GET', 'POST'])
@login_required
def index():
    user = User.query.filter_by(id=session[web_config.FRONT_USER_ID]).first()
    username = user.username
    return render_template('home.html', username=username)


@app.route('/home', methods=['GET', 'POST'])
@login_required
def home():
    user = User.query.filter_by(id=session[web_config.FRONT_USER_ID]).first()
    username = user.username
    return render_template('home.html', username=username)


@app.route('/welcome', methods=['GET'])
@login_required
def welcome():
    return render_template('welcome.html')


# custom management page
@app.route('/train', methods=['GET', 'POST'])
@login_required
def train():

    if request.method == 'POST':
        parameters = {
            'batch_size': request.form.get('batch_size'),
            'local_epoch_num': request.form.get('local_epoch_num'),
            'dataset': request.form.get('dataset'),
            'data_type': request.form.get('data_type'),
            'seed': request.form.get('seed'),
            'optimizer': request.form.get('optimizer'),
            'validation_nodes_num': request.form.get('validation_nodes_num'),
            'flipping_attack_num': request.form.get('flipping_attack_num'),
            'model': request.form.get('model')
        }
        
        main.train(parameters)


    user = User.query.filter_by(id=session[web_config.FRONT_USER_ID]).first()
    username = user.username
    return render_template('train.html', username=username,stop=0)


if __name__ == '__main__':
    app.run(debug=True)
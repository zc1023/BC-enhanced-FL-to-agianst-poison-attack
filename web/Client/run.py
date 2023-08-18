from flask import Flask, request, session, redirect, url_for
from werkzeug.security import generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
import main



app = Flask(__name__)
CORS(app)  # 需要这个库来处理跨源请求
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:////tmp/test.db'  # Use sqlite for demonstration
app.config['SECRET_KEY'] = 'your secret key'
db = SQLAlchemy(app)



def start_training(parameters):
    # 这里是启动联邦学习训练的代码
    train(parameters)

@app.route('/train', methods=['POST'])
def train():
    parameters = request.get_json()  # 从请求中获取参数
    start_training(parameters)
    return 'Training started', 200

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(120), nullable=False)

@app.route('/register', methods=['POST'])
def register():
    username = request.form['username']
    password = request.form['password']
    password_hash = generate_password_hash(password)
    user = User(username=username, password_hash=password_hash)
    db.session.add(user)
    db.session.commit()
    return 'Registered successfully'

@app.route('/login', methods=['POST'])
def login():
    username = request.form['username']
    password = request.form['password']
    user = User.query.filter_by(username=username).first()
    if user is None or not check_password_hash(user.password_hash, password):
        return 'Incorrect username or password'
    session['username'] = username
    return redirect(url_for('index'))

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('index'))

@app.route('/')
def index():
    if 'username' in session:
        return f'Logged in as {session["username"]}'
    return 'You are not logged in'

if __name__ == '__main__':
    db.create_all()  # Create database tables
    app.run(debug=True)

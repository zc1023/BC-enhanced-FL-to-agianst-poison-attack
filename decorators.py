from flask import session,redirect,url_for
from functools import wraps
import web_config

# @wraps(func)，将参数包装起来，避免数据丢失

# 登录验证判断, 看session里有没有ID这一项，如果有说明已经登录了
def login_required(func):
    @wraps(func)
    def inner(*args,**kwargs):
        if web_config.FRONT_USER_ID in session:
           return func(*args,**kwargs)
        else:
            return redirect(url_for('login'))
    return inner

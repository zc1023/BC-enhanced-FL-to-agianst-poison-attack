# BC-enhanced-FL-to-agianst-poison-attack
the code of Chao's bachelor’s dissertation

## Target
1. create 10 nodes and use a `for` loop to complete the simulation of federated learning for ten nodes.

2. baseline is `Fedavg`, `Median`, `Trimmed Mean`, `Bulyan`,  `FoolsGold`, `Krum` and `RFA`.

3. implemrnt node partitioning and training validation process in each round, and complete testing in a non-attack environment (`iid` and `non-iid`).

4. deploy `label flipping` attacks and `Byzatine` attacks.

## algorithm
```python 
for epoch in range(T):
    Tm,Vn = Select(P,m,n)
    for i in Tm:
        get(global model)
        train(i)
        for j in Vn:
            get(mi)
            caculate(ACC(mi(dj)))
    for j in Vn:
        caculate(Dis(mi,global model))
    for i in Tm:
        caculate(score i)
    if i argue:
        caculate(score i)
    Fedavg(model list)
```
## Structure
- src
    - client.py
    - server.py
    - todo
- data
    - mninst
        - client 1
        - client 2
        - ……
- checkpoint
    - epoch1
        - global
        - client 1
        - client 2
        - ……

to do


## Result

### Non-attack
    to do
### label flipping attack
    to do
### byzatine attack
    to do

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
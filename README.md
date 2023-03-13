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

    to do


## Result

### Non-attack
    to do
### label flipping attack
    to do
### byzatine attack
    to do


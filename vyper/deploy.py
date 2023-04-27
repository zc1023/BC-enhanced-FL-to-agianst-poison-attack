# deploy.py  
import json  
from web3 import Web3  
  
# 连接到Sepolia测试网络  
w3 = Web3(Web3.HTTPProvider('https://rpc.sepolia.org'))  
  
# 替换为你的私钥  
private_key = 'ce5c55d2a78dbe8abeba9c862034ec901d68b1caf070f28fa123db5dd5992772'  
  
# 从JSON文件中读取合约ABI和字节码  
with open('average.json') as f:  
    contract_data = f.readlines()
# print(contract_data)
abi = contract_data[0][:-1]
bytecode = bytes.fromhex(contract_data[1][2:-1])
# input(bytecode)
# 获取部署者的地址  
account = w3.eth.account.privateKeyToAccount(private_key).address  
# print(account)
# 获取部署者的nonce  
nonce = w3.eth.getTransactionCount(account)  
# print(nonce)
ChainID = 0xaa36a7
# 创建合约部署事务  
transaction = {  
    'from': account,  
    'gas': int(2100000),  
    'gasPrice': w3.toWei('1', 'gwei'),  
    'nonce': nonce,  
    'data': bytecode,
    'chainId': ChainID
}  
  
# 签署并发送事务  
signed_transaction = w3.eth.account.signTransaction(transaction, private_key)  
# input(signed_transaction.rawTransaction)
transaction_hash = w3.eth.sendRawTransaction(signed_transaction.rawTransaction)  
  
# 获取事务收据  
transaction_receipt = w3.eth.waitForTransactionReceipt(transaction_hash)  
  
# 获取合约地址  
contract_address = transaction_receipt['contractAddress']  
  
print(f"合约已部署到地址: {contract_address}")  


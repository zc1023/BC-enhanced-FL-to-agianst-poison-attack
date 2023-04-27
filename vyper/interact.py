# interact.py  
from web3 import Web3  
import json  
  
# 连接到Sepolia测试网络  
w3 = Web3(Web3.HTTPProvider('https://rpc.sepolia.org'))  
  
# 替换为你的合约地址  
contract_address = '0xEcB115E2FC9ffF8b0161f20d760F19b37eB55d07'  
  
# 从JSON文件中读取合约ABI  
with open('average.json') as f:  
    contract_data = f.readlines()
# print(contract_data)

abi = contract_data[0][:-1]
bytecode = bytes.fromhex(contract_data[1][2:-1])
  
# 创建合约对象  
contract = w3.eth.contract(address=contract_address, abi=abi)  
print(contract)
# 要计算平均值的数组  
numbers = [1,2,3,4,5,
           1,2,3,4,5.0]  
  
# 调用合约的average方法  

average = contract.functions.average(numbers).call()  

print(f"数组 {numbers} 的平均值为: {average}")  


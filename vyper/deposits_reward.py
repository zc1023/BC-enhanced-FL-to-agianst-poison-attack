import json  
from web3 import Web3  
from web3.middleware import geth_poa_middleware  
  
# 将这里的值替换为您的私钥和 Infura 项目 ID  
infura_url = "https://rpc.sepolia.org"  # example: "https://mainnet.infura.io/v3/YOUR-PROJECT-ID"  
private_key = "ce5c55d2a78dbe8abeba9c862034ec901d68b1caf070f28fa123db5dd5992772"  
  
w3 = Web3(Web3.HTTPProvider(infura_url))  
w3.middleware_onion.inject(geth_poa_middleware, layer=0)  # Required for some Ethereum networks  
  
# 设置合约部署者的地址  
deployer_address = w3.eth.account.privateKeyToAccount(private_key).address  
  
# 合约 ABI 和 bytecode（需要从 Vyper 编译器获得）  
contract_abi = [{"stateMutability": "nonpayable", "type": "constructor", "inputs": [], "outputs": []}, {"stateMutability": "payable", "type": "function", "name": "deposit", "inputs": [], "outputs": [{"name": "", "type": "bool"}]}, {"stateMutability": "nonpayable", "type": "function", "name": "distribute_deposits", "inputs": [{"name": "participants", "type": "address[10]"}, {"name": "scores", "type": "uint256[10]"}], "outputs": [{"name": "", "type": "bool"}]}, {"stateMutability": "view", "type": "function", "name": "get_deposit", "inputs": [{"name": "addr", "type": "address"}], "outputs": [{"name": "", "type": "fixed168x10"}]}, {"stateMutability": "view", "type": "function", "name": "contract_creator", "inputs": [], "outputs": [{"name": "", "type": "address"}]}]

contract_bytecode = bytes.fromhex('34610350573360015561033561001a61000039610335610000f36003361161000c5761031d565b60003560e01c63d0e30db0811861009d57600436106103235733604052346402540be4007036f9bfb3af7b756fad5cd10396a21346cb821161032357810290506060526000604051602052600052604060002080546060518082018060140b811861032357905090508155506002546060518082018060140b81186103235790509050600255600160805260206080f35b346103235763d4ebab5381186102c1576102843610610323576004358060a01c610323576040526024358060a01c610323576060526044358060a01c610323576080526064358060a01c6103235760a0526084358060a01c6103235760c05260a4358060a01c6103235760e05260c4358060a01c610323576101005260e4358060a01c6103235761012052610104358060a01c6103235761014052610124358060a01c61032357610160526001543318610323576000610180526000600a905b8060051b61014401356101a052610180516101a05180820182811061032357905090506101805260010181811861015d57505060006101a0526000600a905b8060051b604001516101c0526101a051600981116103235760051b61014401356402540be4007036f9bfb3af7b756fad5cd10396a21346cb821161032357810290506101e052610180516402540be4007036f9bfb3af7b756fad5cd10396a21346cb82116103235781029050610200526002546101e051808202811583838305141715610323576402540be40081058060140b81186103235790509050905061020051801561032357806402540be4008302058060140b81186103235790509050610220526000600060006000610220516402540be4006000821261032357810590506101c0516000f115610323576101a0516001810180600f0b81186103235790506101a05260010181811861019c575050600060025560016101c05260206101c0f35b63f4607feb81186102fc5760243610610323576004358060a01c61032357604052600060405160205260005260406000205460605260206060f35b63bf5c2920811861031b57600436106103235760015460405260206040f35b505b60006000fd5b600080fda165767970657283000307000b005b600080fd')  
  
# 初始化合约对象  
Contract = w3.eth.contract(abi=contract_abi, bytecode=contract_bytecode)  
  
# 部署合约  
transaction = Contract.constructor().buildTransaction({  
    'from': deployer_address,  
    'gas': 500000,  
    'gasPrice':  w3.eth.gasPrice,  
    'nonce': w3.eth.getTransactionCount(deployer_address),  
})  
  
signed_transaction = w3.eth.account.signTransaction(transaction, private_key)  
transaction_hash = w3.eth.sendRawTransaction(signed_transaction.rawTransaction)  
transaction_receipt = w3.eth.waitForTransactionReceipt(transaction_hash)  
contract_address = transaction_receipt['contractAddress']  
  
print(f'合约已部署到地址：{contract_address}')  
  
# 初始化已部署合约的实例  
contract_instance = w3.eth.contract(address=contract_address, abi=contract_abi)  
  
# 交互示例  
  
# 调用 deposit 函数  
transaction = contract_instance.functions.deposit().buildTransaction({  
    'from': deployer_address,  
    'gas': 500000,  
    'gasPrice':  w3.eth.gasPrice,  
    'nonce': w3.eth.getTransactionCount(deployer_address),  
    'value': w3.toWei('0.01', 'ether')  
})  
  
signed_transaction = w3.eth.account.signTransaction(transaction, private_key)  
transaction_hash = w3.eth.sendRawTransaction(signed_transaction.rawTransaction)  
transaction_receipt = w3.eth.waitForTransactionReceipt(transaction_hash)  
  
print(f'deposit 函数调用成功，交易哈希：{transaction_hash.hex()}')  
  
# 调用 get_deposit 函数  
# contract_address = '0x361F78E3f9f3F84774Ac00587385ba4aA7169Ce4'
contract_instance = w3.eth.contract(address=contract_address, abi=contract_abi)  
  
deposit_amount = contract_instance.functions.get_deposit(deployer_address).call()  
print(f'查询到地址 {deployer_address} 的保证金金额：{w3.fromWei(deposit_amount, "ether")} ether')  
  
# 调用 distribute_deposits 函数  
participants = ['0x9b5a496e3525605A5C06510B27e515f3baeBF000'] *10
weights = [111]  *10
  
transaction = contract_instance.functions.distribute_deposits(participants, weights).buildTransaction({  
    'from': deployer_address,  
    'gas': 500000,  
    'gasPrice':  w3.eth.gasPrice,  
    'nonce': w3.eth.getTransactionCount(deployer_address),  
})  
  
signed_transaction = w3.eth.account.signTransaction(transaction, private_key)  
transaction_hash = w3.eth.sendRawTransaction(signed_transaction.rawTransaction)  
transaction_receipt = w3.eth.waitForTransactionReceipt(transaction_hash)  
  
print(f'distribute_deposits 函数调用成功，交易哈希：{transaction_hash.hex()}')  

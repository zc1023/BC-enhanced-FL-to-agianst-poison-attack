from web3 import Web3  
from web3.middleware import geth_poa_middleware  
  
# Replace the values below with your own Ethereum node and private key  
infura_url = "https://rpc.sepolia.org"  # example: "https://mainnet.infura.io/v3/YOUR-PROJECT-ID"  
private_key = "ce5c55d2a78dbe8abeba9c862034ec901d68b1caf070f28fa123db5dd5992772"  
  
w3 = Web3(Web3.HTTPProvider(infura_url))  
w3.middleware_onion.inject(geth_poa_middleware, layer=0)  # Required for some Ethereum networks  
  
# Replace with the ABI generated from the Vyper compiler  
contract_abi = [{"stateMutability": "nonpayable", "type": "function", "name": "upload_array", "inputs": [{"name": "new_array", "type": "uint256[10]"}], "outputs": [{"name": "", "type": "bool"}]}, {"stateMutability": "view", "type": "function", "name": "get_array", "inputs": [], "outputs": [{"name": "", "type": "uint256[10]"}]}]

# Your contract bytecode  
contract_bytecode =  bytes.fromhex('6100e861000f6000396100e86000f36003361161000c576100d0565b60003560e01c346100d65763e065bb2281186100745761014436106100d65760043560005560243560015560443560025560643560035560843560045560a43560055560c43560065560e4356007556101043560085561012435600955600160405260206040f35b63c75d70ed81186100ce57600436106100d65760005460405260015460605260025460805260035460a05260045460c05260055460e052600654610100526007546101205260085461014052600954610160526101406040f35b505b60006000fd5b600080fda165767970657283000307000b')
# Set your account  
my_account = w3.eth.account.privateKeyToAccount(private_key)  
  
contract= w3.eth.contract(bytecode=contract_bytecode, abi=contract_abi)
#building transaction
gas_estimate = w3.eth.estimateGas({'data': contract_bytecode})  
input(gas_estimate)

construct_txn = contract.constructor().buildTransaction({
'from': my_account.address,
'nonce': w3.eth.getTransactionCount(my_account.address),
'gas': gas_estimate*2,
'gasPrice': w3.eth.gasPrice,
'chainId':0xaa36a7,
})
signed = my_account.signTransaction(construct_txn)
tx_hash=w3.eth.sendRawTransaction(signed.rawTransaction)
print(tx_hash.hex())
tx_receipt = w3.eth.waitForTransactionReceipt(tx_hash)
contract_address = tx_receipt['contractAddress']
print("Contract Deployed At:", contract_address )


hash = [1,1,1,0,0,0,0,0,0,0]

contract = w3.eth.contract(address=contract_address, abi=contract_abi)
gas = contract.functions.upload_array(hash).estimateGas({'from':my_account.address})
input(gas)
txn = contract.functions.upload_array(hash).buildTransaction({  
        'from': my_account.address,  
        # 'gas': w3.eth.estimateGas({'to': contract_address, 'from': my_account.address, 'data': contract.encodeABI(contract.functions.submit_array(array_to_submit).fn_name)}),  
        'gas': gas,
        'gasPrice': w3.eth.gasPrice,  
        'nonce': w3.eth.getTransactionCount(my_account.address),  
    })  
signed_txn = my_account.signTransaction(txn)  
txn_hash = w3.eth.sendRawTransaction(signed_txn.rawTransaction)  
txn_receipt = w3.eth.waitForTransactionReceipt(txn_hash)  
data = contract.functions.get_array().call()
print(f'data: {data}')
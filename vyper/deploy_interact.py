from web3 import Web3  
from web3.middleware import geth_poa_middleware  
  
# Replace the values below with your own Ethereum node and private key  
infura_url = "https://rpc.sepolia.org"  # example: "https://mainnet.infura.io/v3/YOUR-PROJECT-ID"  
private_key = "ce5c55d2a78dbe8abeba9c862034ec901d68b1caf070f28fa123db5dd5992772"  
  
w3 = Web3(Web3.HTTPProvider(infura_url))  
w3.middleware_onion.inject(geth_poa_middleware, layer=0)  # Required for some Ethereum networks  
  
# Replace with the ABI generated from the Vyper compiler  
contract_abi = [{"stateMutability": "nonpayable", "type": "constructor", "inputs": [{"name": "_required_submissions", "type": "uint256"}], "outputs": []}, {"stateMutability": "nonpayable", "type": "function", "name": "submit_array", "inputs": [{"name": "array", "type": "uint256[10]"}], "outputs": [{"name": "", "type": "bool"}]}, {"stateMutability": "view", "type": "function", "name": "get_averages", "inputs": [], "outputs": [{"name": "", "type": "uint256[10]"}]}]

  
# Your contract bytecode  
contract_bytecode =  bytes.fromhex('3461024b5760206102506000396000516079556000606e5561022161002961000039610221610000f36003361161000c57610209565b60003560e01c3461020f576384fad77281186101ad57610144361061020f57607954606e5410610044576000604052602060406101ab565b600b606e546009811161020f5702338155600181016004358155602435600182015560443560028201556064356003820155608435600482015560a435600582015560c435600682015560e43560078201556101043560088201556101243560098201555050606e546001810181811061020f579050606e55607954606e54186101a1576000600a905b8060405260006060526000600a905b600b8102805460805260018101805460a052600181015460c052600281015460e05260038101546101005260048101546101205260058101546101405260068101546101605260078101546101805260088101546101a05260098101546101c05250506060516040516009811161020f5760051b60a0015180820182811061020f57905090506060526001018181186100dd575050606051606e54801561020f57808204905090506040516009811161020f57606f01556001018181186100ce5750505b6001604052602060405bf35b63f41dfa448118610207576004361061020f57606f5460405260705460605260715460805260725460a05260735460c05260745460e052607554610100526076546101205260775461014052607854610160526101406040f35b505b60006000fd5b600080fda165767970657283000307000b005b600080fd')
# Set your account  
my_account = w3.eth.account.privateKeyToAccount(private_key)  
  
contract= w3.eth.contract(bytecode=contract_bytecode, abi=contract_abi)
#building transaction
gas_estimate = w3.eth.estimateGas({'data': contract_bytecode})  
input(gas_estimate)

required_submissions = 1
construct_txn = contract.constructor(required_submissions).buildTransaction({
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


# Create contract instance  
contract = w3.eth.contract(address=contract_address, abi=contract_abi)  

# Call the submit_array function  
for i in range(4):
    array_to_submit = [10, 2, 3, 4, 5, 6, 7, 8, 9, 10]  
    submit_array_txn = contract.functions.submit_array(array_to_submit).buildTransaction({  
        'from': my_account.address,  
        # 'gas': w3.eth.estimateGas({'to': contract_address, 'from': my_account.address, 'data': contract.encodeABI(contract.functions.submit_array(array_to_submit).fn_name)}),  
        'gas': contract.functions.submit_array(array_to_submit).estimateGas({'from':my_account.address}),
        'gasPrice': w3.eth.gasPrice,  
        'nonce': w3.eth.getTransactionCount(my_account.address),  
    })  
    signed_submit_array_txn = my_account.signTransaction(submit_array_txn)  
    submit_array_txn_hash = w3.eth.sendRawTransaction(signed_submit_array_txn.rawTransaction)  
    submit_array_txn_receipt = w3.eth.waitForTransactionReceipt(submit_array_txn_hash)  
    
    # Call the get_averages function  
    averages = contract.functions.get_averages().call()  
    print("Averages:", averages)


# array_to_submit = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  
# submit_array_txn = contract.functions.submit_array(array_to_submit).buildTransaction({  
#     'from': my_account.address,  
#     # 'gas': w3.eth.estimateGas({'to': contract_address, 'from': my_account.address, 'data': contract.encodeABI(contract.functions.submit_array(array_to_submit).fn_name)}),  
#     'gas': contract.functions.submit_array(array_to_submit).estimateGas({'from':my_account.address}),
#     'gasPrice': w3.eth.gasPrice,  
#     'nonce': w3.eth.getTransactionCount(my_account.address),  
# })  
# signed_submit_array_txn = my_account.signTransaction(submit_array_txn)  
# submit_array_txn_hash = w3.eth.sendRawTransaction(signed_submit_array_txn.rawTransaction)  
# submit_array_txn_receipt = w3.eth.waitForTransactionReceipt(submit_array_txn_hash)  

# # Call the get_averages function  
# averages = contract.functions.get_averages().call()  
# print("Averages:", averages)

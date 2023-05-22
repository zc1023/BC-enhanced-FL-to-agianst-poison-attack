from web3 import Web3  
from web3.middleware import geth_poa_middleware  
  
# Replace the values below with your own Ethereum node and private key  
infura_url = "https://rpc.sepolia.org"  # example: "https://mainnet.infura.io/v3/YOUR-PROJECT-ID"  
private_key = "ce5c55d2a78dbe8abeba9c862034ec901d68b1caf070f28fa123db5dd5992772"  
  
w3 = Web3(Web3.HTTPProvider(infura_url))  
w3.middleware_onion.inject(geth_poa_middleware, layer=0)  # Required for some Ethereum networks  
  
# Replace with the ABI generated from the Vyper compiler  
contract_abi = [{"name": "StoreIPFSIndex", "inputs": [{"name": "uploader", "type": "address", "indexed": "true"}, {"name": "ipfsIndex", "type": "bytes32", "indexed": "false"}], "anonymous": "false", "type": "event"}, {"stateMutability": "nonpayable", "type": "function", "name": "store_ipfs_index", "inputs": [{"name": "_ipfsIndex", "type": "bytes32"}], "outputs": []}, {"stateMutability": "view", "type": "function", "name": "get_ipfs_index", "inputs": [{"name": "_address", "type": "address"}], "outputs": [{"name": "", "type": "bytes32"}]}]

# Your contract bytecode  
contract_bytecode =  bytes.fromhex('6100c061000f6000396100c06000f36003361161000c576100a8565b60003560e01c346100ae57635a5d7005811861006b57602436106100ae57600435600033602052600052604060002055337f986921cbd28d4c979d4553ff21fae83419ef60e69ac49e33b3658066b14b062960043560405260206040a2005b63868b33bc81186100a657602436106100ae576004358060a01c6100ae57604052600060405160205260005260406000205460605260206060f35b505b60006000fd5b600080fda165767970657283000307000b')
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

contract = w3.eth.contract(address=contract_address, abi=contract_abi)


import torch 
import hashlib

ckpt = torch.load('/home/v-zhoucha/BC-enhanced-FL-to-agianst-poison-attack/log/Exp_iid_CIFAR10_sgd_10_validation_nodes_num_0_flipping_attack_num_0_grad_zero_num_0_grad_scale_num_0_backdoor_num_2/ckpt/iid/0/global.ckpt')
s = hashlib.sha256()
with open('/home/v-zhoucha/BC-enhanced-FL-to-agianst-poison-attack/log/Exp_iid_CIFAR10_sgd_10_validation_nodes_num_0_flipping_attack_num_0_grad_zero_num_0_grad_scale_num_0_backdoor_num_2/ckpt/iid/0/global.ckpt',"rb")as f:
    while b := f.read(8192):
            s.update(b)

hash = s.hexdigest() 

input(hash
      )
gas = contract.functions.store_ipfs_index(hash).estimateGas({'from':my_account.address})
input(gas)
txn = contract.functions.store_ipfs_index(hash).buildTransaction({  
        'from': my_account.address,  
        # 'gas': w3.eth.estimateGas({'to': contract_address, 'from': my_account.address, 'data': contract.encodeABI(contract.functions.submit_array(array_to_submit).fn_name)}),  
        'gas': gas,
        'gasPrice': w3.eth.gasPrice,  
        'nonce': w3.eth.getTransactionCount(my_account.address),  
    })  
signed_txn = my_account.signTransaction(txn)  
txn_hash = w3.eth.sendRawTransaction(signed_txn.rawTransaction)  
txn_receipt = w3.eth.waitForTransactionReceipt(txn_hash)  
data = contract.functions.get_ipfs_index(bytes.fromhex('9b5a496e3525605A5C06510B27e515f3baeBF000')).call()
print(f'data: {data}')

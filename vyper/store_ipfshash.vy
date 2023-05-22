# 声明合约版本  
# vyper_version: 0.2.12  
  
# 定义一个事件以在区块链上记录IPFS索引值（SHA-256哈希值）及上传者地址  
event StoreIPFSIndex:  
    uploader: indexed(address)  
    ipfsIndex: bytes32  
  
# 定义一个映射，用于存储地址和IPFS索引值对  
ipfsIndexes: HashMap[address, bytes32]  
  
# 定义一个公共函数，用于将IPFS索引值（SHA-256哈希值）及上传者地址存储到区块链上，并触发事件  
@external  
def store_ipfs_index(_ipfsIndex: bytes32):  
    self.ipfsIndexes[msg.sender] = _ipfsIndex  
    log StoreIPFSIndex(msg.sender, _ipfsIndex)  
  
# 定义一个公共函数，用于根据地址获取IPFS索引值（SHA-256哈希值）  
@external  
@view  
def get_ipfs_index(_address: address) -> bytes32:  
    return self.ipfsIndexes[_address]  

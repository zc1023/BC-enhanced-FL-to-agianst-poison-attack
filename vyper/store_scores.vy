# Vyper 智能合约代码  
  
# 声明一个长度为 10 的 uint256 型数组  
stored_array: uint256[10]  
  
# 合约函数：上传一个长度为 10 的 uint256 型数组  
@external  
def upload_array(new_array: uint256[10]) -> bool:  
    # 将传入的数组存储到合约的 stored_array 中  
    self.stored_array = new_array  
  
    # 返回 True 表示操作成功  
    return True  
  
# 合约函数：获取存储在合约中的数组  
@external  
@view  
def get_array() -> uint256[10]:  
    # 返回存储在合约中的数组  
    return self.stored_array  

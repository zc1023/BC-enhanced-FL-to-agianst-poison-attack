# Vyper 智能合约代码  
  
# 声明一个地址到 decimal 的映射，用于存储每个参与方的保证金  
deposits: HashMap[address, decimal]  
  
# 设置参与方和权重列表的最大长度  
MAX_PARTICIPANTS: constant(uint256) = 10  
  
# 添加合约创建者变量并使用 public 修饰符声明  
contract_creator: public(address)  
  
# 添加保证金池变量  
total_deposit_pool: decimal  
  
# 合约构造函数：设置合约创建者  
@external  
def __init__():  
    self.contract_creator = msg.sender  
  
# 合约函数：参与方押入保证金  
@external  
@payable  
def deposit() -> bool:  
    # 获取发送者地址和押入的保证金金额  
    sender: address = msg.sender  
    amount: decimal = convert(msg.value, decimal)  
  
    # 将保证金存储到映射中  
    self.deposits[sender] += amount  
  
    # 将保证金添加到保证金池中  
    self.total_deposit_pool += amount  
  
    # 返回 True 表示操作成功  
    return True 
  
# 合约函数：根据输入的分数列表按比例分配保证金池  
@external  
def distribute_deposits(participants: address[MAX_PARTICIPANTS], scores: uint256[MAX_PARTICIPANTS]) -> bool:  
    # 检查合约调用者是否为合约的创建者  
    assert msg.sender == self.contract_creator  
  
    # 计算分数总和  
    total_score: uint256 = 0
    for score in scores:  
        total_score += score  
  
    # 遍历参与方列表，按分数分配保证金池  
    idx: int128 = 0  
    for participant in participants:  
        # 计算按分数分配的保证金池金额  
        decimal_score: decimal = convert(scores[idx], decimal)  
        decimal_total_score: decimal = convert(total_score, decimal)  
        distributed_amount: decimal = self.total_deposit_pool * decimal_score / decimal_total_score  
  
        # 发放分配后的保证金  
        send(participant, convert(distributed_amount, uint256))  
  
        # 更新索引  
        idx += 1  
  
    # 清空保证金池  
    self.total_deposit_pool = convert(0, decimal)  
  
    # 返回 True 表示操作成功  
    return True  
  
# 合约函数：查询指定地址的保证金金额  
@external  
@view  
def get_deposit(addr: address) -> decimal:  
    # 返回指定地址的保证金金额  
    return self.deposits[addr]  

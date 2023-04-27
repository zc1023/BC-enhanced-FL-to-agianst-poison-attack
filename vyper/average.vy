# average.vy  

@external  
def average(numbers: int8[10]) -> int8:  
    total: int8 =0 
    for i in numbers:  
        total += i  
    return total / 10

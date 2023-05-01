# Vyper smart contract for array submission and averaging  
  
# Define the array structure  
struct OneDimensionalArray:  
    submitter: address  
    array: uint256[10]  
  
# Define the storage variables  
arrays: OneDimensionalArray[10]  
submission_count: uint256  
averages: uint256[10]  
required_submissions: uint256  
  
@external  
def __init__(_required_submissions: uint256):  
    self.required_submissions = _required_submissions  
    self.submission_count = 0  
  
@external  
def submit_array(array: uint256[10]) -> bool:  
    # Check if enough submissions have already been received  
    if self.submission_count >= self.required_submissions:  
        return False  
  
    # Store the submitted array  
    self.arrays[self.submission_count] = OneDimensionalArray({  
        submitter: msg.sender,  
        array: array  
    })  
      
    self.submission_count += 1  
  
    # Check if enough submissions have been received, and calculate averages if needed  
    if self.submission_count == self.required_submissions:  

        for i in range(10):
            total: uint256 = 0  
            for participate in self.arrays:  
                total += participate.array[i]  
            self.averages[i] = total / self.submission_count
  
    return True  

@external
@view  
def get_averages() -> uint256[10]:  
    return self.averages  

storedData: public(bytes32[514])

# data: HashMap[address, bytes32]
event Datasubmitted:
    data: bytes32[514]


@external
def set(_x: bytes32[514]):
    self.storedData = _x
    log Datasubmitted(_x)

@external
@view
def get() -> bytes32[514]:
    return self.storedData
from ecdsa import SigningKey, VerifyingKey, SECP256k1

def get_uncompressed_pubkey(pubkey):
    return '04' + pubkey.to_string().hex()

def verify_key_pair(private_key, uncompressed_public_key):
    try:
        # 从字节数据创建签名密钥对象
        sk = SigningKey.from_string(bytes.fromhex(private_key), curve=SECP256k1)
        
        # 从未压缩公钥创建验证密钥对象
        if uncompressed_public_key.startswith('04'):
            vk = VerifyingKey.from_string(bytes.fromhex(uncompressed_public_key[2:]), curve=SECP256k1)
        else:
            return False

        # 生成签名
        message = b"Hello, world!"  # 用于签名和验证的任意消息
        signature = sk.sign(message)

        # 验证签名
        return vk.verify(signature, message)
    except:
        return False

# 生成私钥
sk = SigningKey.generate(curve=SECP256k1)

# 从私钥获取公钥
vk = sk.get_verifying_key()

# 获取未压缩的公钥
uncompressed_pubkey = get_uncompressed_pubkey(vk)

# 打印私钥和未压缩的公钥
private_key = sk.to_string().hex()
print("Private Key:", private_key)
print("Uncompressed Public Key:", uncompressed_pubkey)

# 验证公钥和私钥是否匹配
if verify_key_pair(private_key, uncompressed_pubkey):
    print("公钥和私钥匹配！")
else:
    print("公钥和私钥不匹配。")

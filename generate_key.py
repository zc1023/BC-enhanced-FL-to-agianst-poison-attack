from ecdsa import SigningKey, VerifyingKey, SECP256k1

def get_compressed_pubkey(pubkey):
    coordinates = pubkey.to_string()
    x = coordinates[:len(coordinates)//2]
    y = coordinates[len(coordinates)//2:]
    if int.from_bytes(y, byteorder='big', signed=False) % 2 == 0:
        return '02' + x.hex()
    else:
        return '03' + x.hex()

def verify_key_pair(private_key, compressed_public_key):
    try:
        sk = SigningKey.from_string(bytes.fromhex(private_key), curve=SECP256k1)
        vk = VerifyingKey.from_encoded_point(bytes.fromhex(compressed_public_key), curve=SECP256k1)

        message = b"Hello, world!"
        signature = sk.sign(message)
        return vk.verify(signature, message)
    except Exception as e:
        print("Error:", e)  # 输出错误，以帮助我们确定问题
        return False

sk = SigningKey.generate(curve=SECP256k1)
vk = sk.get_verifying_key()

compressed_pubkey = get_compressed_pubkey(vk)

private_key = sk.to_string().hex()
print("Private Key:", private_key)
print("Compressed Public Key:", compressed_pubkey)

if verify_key_pair(private_key, compressed_pubkey):
    print("公钥和私钥匹配！")
else:
    print("公钥和私钥不匹配。")

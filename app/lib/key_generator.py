from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

# 1. Generate private key
private_key = rsa.generate_private_key(
    public_exponent=65537,
    key_size=2048
)

# 2. Convert to PEM format (Private)
pem_private = private_key.private_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PrivateFormat.PKCS8,
    encryption_algorithm=serialization.NoEncryption()
)

# 3. Convert to PEM format (Public)
pem_public = private_key.public_key().public_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PublicFormat.SubjectPublicKeyInfo
)

# 4. Print out for pasting or saving directly
env_ready_private_key = pem_private.decode('utf-8').replace('\n', '\\n')
env_ready_public_key = pem_public.decode('utf-8').replace('\n', '\\n')
print("=== PRIVATE KEY ===")
print(env_ready_private_key)
print("\n=== PUBLIC KEY ===")
print(env_ready_public_key)
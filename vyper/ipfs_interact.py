from ipyfs import Files,FileStore

files = Files(
    host="http://localhost",  # Set IPFS Daemon Host
    port=5001  # Set IPFS Daemon Port
)
filestore = FileStore(
)

# Read the file and upload it to IPFS.
with open("helloworld.json", "rb") as f:
    files.write(
        path=f"/{f.name}",
        file=f,
        create=True
    )

# Get the information of the uploaded file.
info = files.stat('/helloworld.json')
print(info)

filestore.ls('QmScce4G7C8p1MrQkNW4PgvetiSss8Zzj7gMD6CsM5PeRT')
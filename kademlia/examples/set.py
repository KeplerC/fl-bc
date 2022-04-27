import logging
import asyncio
import sys
import os 
import lzma
import pickle

from kademlia.network import Server

if len(sys.argv) != 5:
    print("Usage: python set.py <bootstrap node> <bootstrap port> <key> <value>")
    sys.exit(1)

handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
log = logging.getLogger('kademlia')
log.addHandler(handler)
log.setLevel(logging.DEBUG)

async def run():
    server = Server()
    await server.listen(8469)
    bootstrap_node = (sys.argv[1], int(sys.argv[2]))
    await server.bootstrap([bootstrap_node])
    
    glob_file_name = sys.argv[3]
    
    with open(glob_file_name, "rb") as f:
        lzma_w = f.read()

    # load file 
    # while not os.path.exists(glob_file_name):
    #     time.sleep(1)
    # if os.path.isfile(glob_file_name):
    #     open_file = lzma.open(glob_file_name,'rb')
    #     lzma_w = pickle.load(open_file)
    #     open_file.close()
    # else:
    #     raise ValueError("%s isn't a file!" % glob_file_name)


    # compressor = lzma.LZMACompressor()
    # compressed = compressor.compress(lzma_w)
    # compressed = compressor.compress(b"hello")

    # format:
    # mapping:
    # name : num_segment
    # name_x: the x_th segment 

    num_segment = len(lzma_w) // 8000
    print("Number of segment is : ", num_segment)
    segments = []
    for i in range(num_segment - 1):
        segments.append(lzma_w[i * 8000: (i+1) * 8000])
    segments.append(lzma_w[num_segment * 8000:])

    await server.set(sys.argv[3], str(num_segment))
    for i_segment in range(len(segments)):
        print(f"sent segment {i_segment} with length {len(segments[i_segment])}")
        await server.set(sys.argv[3] + "_" + str(i_segment),segments[i_segment])
        
    #print("type of lzma_w", type(lzma_w))
    #print("lzma_w size::::: ", sys.getsizeof(lzma_w))
    print("glob_file_name " , sys.argv) 

    server.stop()

asyncio.run(run())
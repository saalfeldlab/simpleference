import socket


class BlockClient(object):
    request = "1"

    def __init__(self, host, port):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((host, port))

    # request a block (single word str)
    def request_block(self):
        self.sock.sendall(bytes(self.request + '\n', 'utf-8'))
        block_offset = str(self.sock.recv(1024), 'utf-8')
        block_offset = [int(bo) for bo in block_offset]
        return block_offset

    def confirm_block(self, block_offset):
        assert len(block_offset) == 3
        request = " ".join(block_offset)
        self.sock.sendall(bytes(request + '\n', 'utf-8'))
        success = str(self.sock.recv(1024), 'utf-8')
        return success

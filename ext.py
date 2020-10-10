import os

from time import ctime

from pickle import load, dump, HIGHEST_PROTOCOL

from multiprocessing import Pool, cpu_count
from multiprocessing.pool import ThreadPool as Pool2


##


def cls():
    os.system('clear' if os.name != 'nt' else 'cls')


##


def now():
    return [e for e in ctime().split(' ') if ':' in e][-1]


##


def pickle_save(obj, file_path, buffered=False):
    with open(file_path, 'wb+') as f:
        if buffered:
            return dump(obj, BufferedFile(f), protocol=HIGHEST_PROTOCOL)
        else: return dump(obj, f)

def pickle_load(file_path, buffered=False):
    try:
        with open(file_path, 'rb') as f:
            if buffered:
                return load(BufferedFile(f))
            else: return load(f)
    except Exception as e:
        return None


##


class BufferedFile(object):

    def __init__(self, f):
        self.f = f

    def __getattr__(self, item):
        return getattr(self.f, item)

    def read(self, n):
        if n >= (1 << 31):
            buffer = bytearray(n)
            i = 0
            while i < n:
                batch_size = min(n - i, 1 << 31 - 1)
                buffer[i:i + batch_size] = self.f.read(batch_size)
                i += batch_size
            return buffer
        return self.f.read(n)

    def write(self, buffer):
        n = len(buffer)
        i = 0
        while i < n:
            batch_size = min(n - i, 1 << 31 - 1)
            self.f.write(buffer[i:i + batch_size])
            i += batch_size


##


def parallel(fn,lst,chunksize=None,backend='proc',hm_workers=None):
    if not hm_workers: hm_workers = cpu_count()
    with (Pool if backend=='proc' else Pool2)(hm_workers) as p:
        res = p.map_async(fn,lst,chunksize)
        p.close()
        p.join()
        return res.get()


##
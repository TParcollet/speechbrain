import multiprocessing
import functools


def line_func_to_chunk_func(func):
    @functool.wraps(func)
    def wrapper(line_iterator):
        output = []
        for line in iterator:
            output.append(func(line))
        return output

    return wrapper


def process_file_in_parallel_chunks(filepath, num_workers, chunk_func):
    with multiprocessing.Pool(num_workers):
        pass


def _chunkify(filepath, num_chunks):
    pass

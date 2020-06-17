import argparse


def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size')
    parser.add_argument('--n_threads', default=8, type=int, help='Number of threads for multi-thread loading')
    parser.add_argument('--drop_last', default=True, type=bool, help='drop the remaining of the batch if the size doesnt match minimum batch size')
    args = parser.parse_args()
    return args

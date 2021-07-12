import argparse
import subprocess as sp
import re
import time

from loguru import logger

PYTHON="/lila/home/schmidth/.conda/envs/bio/bin/python3.8"
bsub_args = [
    '-o', 'bsub_files/%J.stdout',
    '-eo', 'bsub_files/%J.stderr',
    '-q', 'gpuqueue',
    '-gpu', '-',
    '-R' , 'rusage[mem=8GB]',
    '-W', '48:00'
]

def get_host(job_id):
    args = ["bjobs", "-l", str(job_id)]
    process = sp.run(args, capture_output=True, text=True)
    matches = re.search(r'Host\(s\) <([^>]+)>', process.stdout)
    if matches is None:
        return None
    return matches[1]

def submit_bsub_job(args):
    '''Submits a bsub job and returns its ID'''
    args = ['bsub'] + args
    process = sp.run(args, capture_output=True, text=True)
    job_id = int(re.match(r'Job <([0-9]+)>', process.stdout)[1])
    return process, job_id
    
def get_job_arguments(args, rank, master=None):
    cmd_args = bsub_args
    cmd_args += ['-J', 'be-ml-master' if rank == 0 else f'be-ml-rank-{rank}']
    py_cmd_args = [
        args.training_data,
        '-e', args.epochs,
        '-r', rank,
        '-n', args.num_processes,
        '--keep-best',
    ]

    if args.test_data is not None:
        py_cmd_args += ['-t', args.test_data]

    if args.checkpoint is not None:
        py_cmd_args += ['-c', args.checkpoint]

    if args.save is not None:
        py_cmd_args += ['-s', args.save]

    if args.load_checkpoint is not None:
        py_cmd_args += ['-l', args.load_checkpoint]

    if master is not None:
        py_cmd_args += ['-m', master]

    py_cmd_args = list(map(str, py_cmd_args))

    cmd = (
        'module load nccl/2.8.3-cuda10.2; ' +
        'NCCL_IB_DISABLE=1 ' + 
        f'{PYTHON} scripts/model.py train {" ".join(py_cmd_args)}'
    )

    return cmd_args + [cmd]

def start_master(args):
    '''Start master process, returning host and BSUB job id.'''
    master_args = get_job_arguments(args, 0)
    proc, job_id = submit_bsub_job(master_args)
    logger.info('Started master process.')
    while (host := get_host(job_id)) is None:
        logger.info('Polling master process.')
        time.sleep(1)
    return host

def spawn_processes(args):
    host = start_master(args)
    host = f'{host}:25000'
    for rank in range(1, args.num_processes):
        logger.info(f'Starting node with rank {rank}')
        proc_args = get_job_arguments(args, rank, master=host)
        proc, job_id = submit_bsub_job(proc_args)

def parse_args():
    parser = argparse.ArgumentParser(
        prog='base-editing-ml',
        description='Distributed GPU training.'
    )

    parser.add_argument(
        'training_data',
        help='CSV file that contains training data',
    )
    parser.add_argument(
        '-t', '--test-data',
        help='CSV file thta contains test data.'
    )
    parser.add_argument(
        '-c', '--checkpoint',
        help='File to store intermediary training state',
    )
    parser.add_argument(
        '-e', '--epochs',
        help='Number of epochs to run',
        type=int, default=200, 
    )
    parser.add_argument(
        '-s', '--save',
        help='File to store trained model',
    )
    parser.add_argument(
        '-l', '--load-checkpoint',
        help='File containing stored training state',
    )
    parser.add_argument(
        '-n', '--num_processes',
        help='Number of processes',
        default=1,
        type=int
    )

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    spawn_processes(args)

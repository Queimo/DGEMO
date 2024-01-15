import ray
import argparse
import os
import signal
from time import time, sleep
from main import run_experiment
# from baselines.nsga2 import run_experiment as run_experiment_nsga2
from arguments import extract_args
import shlex
from datetime import datetime
import gc

MAX_NUM_PENDING_TASKS = 8


@ray.remote(num_cpus=2)
def worker(cmd, problem, algo, seed, datetime_str):
    cmd_args = shlex.split(cmd)
    cmd_args = cmd_args[2:]
    args, framework_args = extract_args(cmd_args)
    
    framework_args["datetime_str"] = datetime_str

    start_time = time()
    
    if algo == 'nsga2':
        # run_experiment_nsga2(args, framework_args)
        pass
    else:
        try:
            run_experiment(args, framework_args)
        except Exception as e:
            print(e)
            print(f'problem {problem} algo {algo} seed {seed} failed, time: {time() - start_time:.2f}s')
            return 0, problem, algo, seed
        
    runtime = time() - start_time
    
    return runtime, problem, algo, seed

def main():
    ray.init()
    parser = argparse.ArgumentParser()
    parser.add_argument('--problem', type=str, nargs='+', required=True, help='problems to test')
    parser.add_argument('--algo', type=str, nargs='+', required=True, help='algorithms to test')
    parser.add_argument('--n-seed', type=int, default=8, help='number of different seeds')
    parser.add_argument('--n-process', type=int, default=os.cpu_count(), help='number of parallel optimization executions')
    parser.add_argument('--n-inner-process', type=int, default=1, help='number of process can be used for each optimization')
    parser.add_argument('--subfolder', type=str, default='default', help='subfolder of result')
    parser.add_argument('--exp-name', type=str, default=None, help='custom experiment name')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--n-iter', type=int, default=8)
    parser.add_argument('--n-init-sample', type=int, default=5, help='number of initial design samples')
    parser.add_argument('--n-var', type=int, default=2)
    parser.add_argument('--n-obj', type=int, default=2)
       
    args = parser.parse_args()

    start_time = time()
    tasks = []
    
    datetime_str = datetime.now().strftime("%Y%m%d-%H%M%S")

    for seed in range(args.n_seed):
        for problem in args.problem:
            for algo in args.algo:

                if algo == 'nsga2':
                    command = f'python baselines/nsga2.py \
                        --problem {problem} --seed {seed} \
                        --batch-size {args.batch_size} --n-iter {args.n_iter} \
                        --n-process {args.n_inner_process} \
                        --subfolder {args.subfolder} --log-to-file'
                else:
                    command = f'python main.py \
                        --problem {problem} --algo {algo} --seed {seed} \
                        --batch-size {args.batch_size} --n-iter {args.n_iter} \
                        --n-process {args.n_inner_process} \
                        --n-init-sample {args.n_init_sample} \
                        --subfolder {args.subfolder} --log-to-file'
                    if algo != 'dgemo':
                        command += ' --n-gen 200'

                command += f' --n-var {args.n_var} --n-obj {args.n_obj}'

                if args.exp_name is not None:
                    command += f' --exp-name {args.exp_name}'

                if len(tasks) > MAX_NUM_PENDING_TASKS:
                    completed_tasks, tasks = ray.wait(tasks, num_returns=1)
                    runtime, ret_problem, ret_algo, ret_seed = ray.get(completed_tasks[0])
                    if runtime != 0:
                        print(f'problem {ret_problem} algo {ret_algo} seed {ret_seed} done, time: {time() - start_time:.2f}s, runtime: {runtime:.2f}s')

                sleep(1)
                task = worker.remote(command, problem, algo, seed, datetime_str)
                tasks.append(task)
                print(f'problem {problem} algo {algo} seed {seed} started')
                gc.collect()
    
    while len(tasks) > 0:
        completed_tasks, tasks = ray.wait(tasks, num_returns=1)
        runtime, ret_problem, ret_algo, ret_seed = ray.get(completed_tasks[0])
        if runtime != 0:
            print(f'problem {ret_problem} algo {ret_algo} seed {ret_seed} done, time: {time() - start_time:.2f}s, runtime: {runtime:.2f}s')
    

    print('all experiments done, time: %.2fs' % (time() - start_time))

if __name__ == "__main__":
    main()

import ray
import argparse
import os
import signal
from time import time
from get_ref_point import get_ref_point
import logging

@ray.remote
def worker(cmd, problem, algo, seed):
    from main_kwargs import run_experiment
    from arguments import extract_args
    import shlex
    cmd_args = shlex.split(cmd)
    cmd_args = cmd_args[2:]
    args, framework_args = extract_args(cmd_args)

    run_experiment(args, framework_args)
    ret_code = 0
    return ret_code, problem, algo, seed

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--problem', type=str, nargs='+', required=True, help='problems to test')
    parser.add_argument('--algo', type=str, nargs='+', required=True, help='algorithms to test')
    parser.add_argument('--n-seed', type=int, default=8, help='number of different seeds')
    parser.add_argument('--n-process', type=int, default=1, help='number of parallel optimization executions')
    parser.add_argument('--n-inner-process', type=int, default=1, help='number of process can be used for each optimization')
    parser.add_argument('--subfolder', type=str, default='default', help='subfolder of result')
    parser.add_argument('--exp-name', type=str, default=None, help='custom experiment name')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--n-iter', type=int, default=20)
    parser.add_argument('--n-var', type=int, default=2)
    parser.add_argument('--n-obj', type=int, default=2)

    args = parser.parse_args()

    ray.init(local_mode=False, num_cpus=args.n_process,logging_level=logging.DEBUG
             )
    ref_dict = {}
    for problem in args.problem:
        ref_point = get_ref_point(problem, args.n_var, args.n_obj)
        ref_point_str = ' '.join([f' " {str(val)}"' for val in ref_point])
        ref_dict[problem] = ref_point_str

    start_time = time()
    tasks = []

    for seed in range(args.n_seed):
        for problem in args.problem:
            for algo in args.algo:
                # [Build your command string here]d

                if algo == 'nsga2':
                    command = f'python baselines/nsga2.py \
                        --problem {problem} --seed {seed} \
                        --batch-size {args.batch_size} --n-iter {args.n_iter} \
                        --ref-point {ref_dict[problem]} \
                        --n-process {args.n_inner_process} \
                        --subfolder {args.subfolder} --log-to-file'
                else:
                    command = f'python main.py \
                        --problem {problem} --algo {algo} --seed {seed} \
                        --batch-size {args.batch_size} --n-iter {args.n_iter} \
                        --ref-point {ref_dict[problem]} \
                        --n-process {args.n_inner_process} \
                        --subfolder {args.subfolder} --log-to-file'
                    if algo != 'dgemo':
                        command += ' --n-gen 200'

                command += f' --n-var {args.n_var} --n-obj {args.n_obj}'

                if args.exp_name is not None:
                    command += f' --exp-name {args.exp_name}'

                # Start the worker as a Ray remote function
                task = worker.remote(command, problem, algo, seed)
                tasks.append(task)
                print(f'problem {problem} algo {algo} seed {seed} added to queue')
    
    print(f'{len(tasks)} tasks in total')
    while len(tasks) > 0:
        ready_refs, remaining_refs = ray.wait(tasks, num_returns=1, timeout=None)
        tasks = remaining_refs
        print(f'{len(tasks)} tasks remaining')
        print(f'finished {ready_refs}')
    
    print('all experiments done, time: %.2fs' % (time() - start_time))

if __name__ == "__main__":
    main()



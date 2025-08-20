import pandas as pd
import csv
import subprocess
import shlex
import time
import os
import random
from concurrent.futures import ThreadPoolExecutor
import typer # type: ignore
import numpy as np

import subprocess
import shlex
import numpy as np


def next_avai_gpu(gpu_alloc, mode="standard"):
    """
    Function to check if specified GPUs are idle based on their memory usage.

    :param gpu_alloc: String of comma-separated GPU indices to check,
    :return: True if all specified GPUs are idling, False otherwise.

    are_gpus_idle ("0 1")
    are_gpus_idle ("0 1 2 3")
    """
    # Convert comma-separated list of GPUs into a list
    # gpu_alloc = gpu_alloc[1:-1]
    gpu_indices = gpu_alloc.split(" ")

    # Use `nvidia-smi` to get the memory used by the specified GPU
    # mem_cmd = f"nvidia-smi --query-gpu=index,memory.used --format=csv,noheader"
    # mem_result = subprocess.run(shlex.split(cmd), stdout=subprocess.PIPE, text=True)
    # mem_lines = result.stdout.strip().split('\n')

    cmd = f"nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader"
    result = subprocess.run(shlex.split(cmd), stdout=subprocess.PIPE, text=True)
    lines = result.stdout.strip().split("\n")

    # print (lines)
    gpu_mem = {}
    gpu_util = {}
    # ---------------------------
    # Random shuffle the order of next avai gpu
    # ---------------------------

    for gpu_index in gpu_indices:
        gpu_status = None
        for line in lines:
            index, memory_used, utilization = line.split(", ")
            if index == gpu_index:
                gpu_mem_status = int(
                    memory_used.split(" ")[0]
                )  # Extract memory usage and convert to int
                gpu_utilization_status = int(
                    memory_used.split(" ")[0]
                )  # Extract memory usage and convert to int
                break

        if gpu_mem_status is None:
            print(f"GPU index {gpu_index} not found.")
            return False

        # print(gpu_mem_status)
        gpu_mem[int(gpu_index)] = gpu_mem_status
        gpu_util[int(gpu_index)] = gpu_utilization_status

    min_idx, min_mem = min(gpu_mem.items(), key=lambda x: x[1])

    if mode == "standard":
        # Check if GPU memory usage is below the threshold (10000 MB)
        if min_mem <= 60000:
            # print (f'found idling gpu at cuda {gpu_index}')
            return min_idx
    elif mode == "greedy":
        if (min_mem <= 15000) | (gpu_util[min_idx] < 100):
            # print (f'found idling gpu at cuda {gpu_index}')
            return min_idx
    else:
        raise Exception("no such mode found in next_avai_gpu")

    # print("No idling GPU found")
    return None


def found_exception(serie):
    """
    Return True if if found exception for this scheduled run
            False otherwise
    """
    itr_id = serie["iteration_ids"]
    # Check if the folder exists at the end of each iteration
    folder_path = f"./logs/exception/{itr_id}.log"  # Specify the folder path
    # print(f"checking {folder_path}")
    if os.path.exists(folder_path):
        return True
    return False


def is_process_running(serie):
    """
    check if the given requested run is running.
    It will call children function is_specified_running with specified keywords

    :param: serie for requested parameters
    :return: True is running, False otherwise
    """

    check_method = f'method {serie["method"]}'
    check_bit = f'task_length {serie["task_length"]}'
    check_pu = ""
    check_seed = f'{serie["data_seed"]}'
    if serie["processing_unit"] == "cpu":
        check_pu = "data.py"
    elif serie["processing_unit"] == "gpu":
        check_pu = "scaling_law_exp"
    else:
        raise Exception("no specified processing unit")
    check_seed

    check_method = f'method {serie["method"]}'

    if found_exception(serie):
        itr_ids = serie["iteration_ids"]
        raise Exception(f"Exception found in ./logs/exception/{itr_ids}.log")

    if (
        (not is_specified_running(check_method))
        | (not is_specified_running(check_bit))
        | (not is_specified_running(check_pu))
        | (not is_specified_running(check_seed))
    ):
        return False
    else:
        return True


def is_specified_running(process_name):
    """
    Check if a process with the given name is currently running.

    :param process_name: String of the process name to look for
    :return: True if found related prodess running, False otherwise

    is_process_running('method cot_c5')
    is_process_running('model_dim 24')
    is_process_running('task_length 15')
    """
    try:
        # This command searches for the process by name and returns its PID if found.
        subprocess.run(
            ["pgrep", "-f", process_name], check=True, stdout=subprocess.PIPE
        )
        # print (f'found process_name={process_name}')
        return True
    except subprocess.CalledProcessError:
        return False


def run_experiment(serie, gpu_id):
    """
    Run the given experiment with the given parameters. The command is run in the background.

    :param:
    :serie: serie that follows the CSV FORMAT
    :gpu_id: the index of gpu running this job

    :return: the subprocess id

    start_experiment (df.iloc[0])
    start_experiment (df.iloc[2])
    """

    # Major identities
    experiment = serie["experiment"]
    task_length = serie["task_length"]
    number_of_nodes = serie["number_of_nodes"]
    method = serie["method"]
    ablations = serie["ablations"]
    model_dim = serie["model_dim"]
    num_layers = serie["num_layers"]
    num_heads = serie["num_heads"]
    task = serie["task"]
    lr = serie["lr"]
    iteration_ids = serie["iteration_ids"]
    repetitions = serie["repetitions"]

    # batch_size
    train_batch_size = serie["train_batch_size"]
    test_batch_size = serie["test_batch_size"]

    # thres and allocation
    # max_num_samples = serie["max_num_samples"]
    initial_compute_estimate = serie["est_compute"]

    # sample size
    test_set_size = serie["test_set_size"]
    val_set_size = serie["val_set_size"]

    # seeds
    data_seed = serie["data_seed"]
    model_seed = serie["model_seed"]

    input_reverse = serie["input_reverse"]
    output_reverse = serie["output_reverse"]
    skip_line = serie["skip_line"]
    threshold_testing = serie["threshold_testing"]
    early_stopping = serie["early_stopping"]

    # directory
    # dataset_dir = serie['dataset_dir']
    dataset_dir = f"./datasets/{task}/{repetitions}/{task_length}/{method}/{ablations}/{int(lr * 1e4)}/{input_reverse}/{output_reverse}/{skip_line}/{model_dim}/{num_layers}/{num_heads}/{data_seed}/{model_seed}"
    train_dataset_filepath = f"{dataset_dir}/train_dataset"
    val_dataset_filepath = f"{dataset_dir}/val_dataset"
    test_dataset_filepath = f"{dataset_dir}/test_dataset"

    # others
    loss_denom = serie["loss_denom"]
    precision = serie["precision"]

    # log
    train_logdir = "./logs/runtimes/train"
    data_logdir = "./logs/runtimes/data"
    os.makedirs(train_logdir, exist_ok=True)
    os.makedirs(data_logdir, exist_ok=True)
    data_logpath = f"{data_logdir}/{task_length}bits-{method}-modeldim_{model_dim}-gpu{gpu_id}-data-seed_{data_seed}.log"
    train_logpath = f"{train_logdir}/{task_length}bits-{method}-modeldim_{model_dim}-gpu{gpu_id}-data-seed_{data_seed}.log"

    stats_dirpath = serie["stats_dirpath"]

    if experiment == "run_experiments.sh":
        # print ("inside experiment == run_experiments.sh")

        if serie["processing_unit"] == "cpu":
            if "graph" in task:
                command = (
                    f"sage --python3 data.py --task_length {task_length} "  #  --max_num_samples {max_num_samples}
                    f"--test_set_size {test_set_size} --val_set_size {val_set_size} --seed {data_seed} --task {task} --model_dim {model_dim} --num_layer {num_layers} --compute_budget {initial_compute_estimate} "
                    f"--method {method} --ablations {ablations} --skip_line {skip_line} --dataset_dir {dataset_dir} {'--input_reverse' if input_reverse else ''} {'--output_reverse' if output_reverse else ''} --repetitions {repetitions} {f'--number_of_nodes {number_of_nodes}' if number_of_nodes else ''} > {data_logpath}"
                )
            else:
                command = (
                    f"python3 data.py --task_length {task_length} "  #  --max_num_samples {max_num_samples}
                    f"--test_set_size {test_set_size} --val_set_size {val_set_size} --seed {data_seed} --task {task} --model_dim {model_dim} --num_layer {num_layers} --compute_budget {initial_compute_estimate} "
                    f"--method {method} --ablations {ablations} --skip_line {skip_line} --dataset_dir {dataset_dir} {'--input_reverse' if input_reverse else ''} {'--output_reverse' if output_reverse else ''} --repetitions {repetitions} {f'--number_of_nodes {number_of_nodes}' if number_of_nodes else ''} > {data_logpath}"
                )
            print(command)
            return subprocess.Popen(command, shell=True)

        elif serie["processing_unit"] == "gpu":
            # Construct the command
            command = (
                f"CUDA_VISIBLE_DEVICES={gpu_id} python3 scaling_law_exp.py "
                f"--seed {model_seed} --method {method} --ablations {ablations} "
                f"--test_batch_size_scaling {test_batch_size} --train_batch_size {train_batch_size} --learning_rate {lr} "
                f"--loss_denom {loss_denom} --task_length {task_length} --task {task} --repetitions {repetitions} {'--early_stopping' if early_stopping else ''} {'--threshold_testing' if threshold_testing else ''} "
                f"--initial_compute_estimate {initial_compute_estimate} --model_dimensions {model_dim} --num_layer {num_layers} --num_head {num_heads} "
                f"--iteration_ids {iteration_ids} --train_dataset_filepath {train_dataset_filepath} "
                f"--test_dataset_filepath {test_dataset_filepath} --val_dataset_filepath {val_dataset_filepath} "
                f"--stats_dirpath {stats_dirpath} "
                f"--precision {precision} > {train_logpath}"
            )
            print(command)
            # print ("inside allocating gpu")
            return subprocess.Popen(command, shell=True)
            # return True
    elif experiment == "run_cbs_measurements.sh":
        raise Exception("run_cbs_measurements.sh not implemented yet")
    else:
        raise Exception(f"experiment not found: {experiment}")

    # command = f"./{experiment} --task_length {bit} --method {scenario} --model_dims {model_dim} --gpu_allocation {gpu_alloc} --learning_rates {lr} --train_batch_size {train_batch} --max_num_samples 4000000 --initial_compute_estimate {compute_resource} --iteration_ids {iteration_ids}"
    # print(command)

    # Start the experiment in the background and return the process
    return subprocess.Popen(command, shell=True)


def scheduling_runs(df, gpu_alloc, mode="standard", debug=False, profiling=True):
    """
    This function will iteratively run all requested jobs in given df under specified mode and gpu_alloc

    param:
    df - all specified runs, where each serie specify all necessary arguments for 1 run
    gpu_alloc - specify all the gpus that are allocated
    mode - standard: only look for gpus that are specifically allocated
         - greedy: will iteratively look for all available gpus and run on them disregarding gpu_alloc

    return: None

    scheduling_runs (df, "0,1,2,3")
    scheduling_runs (df, "0,1,2,3", mode='standard')
    scheduling_runs (df, "3,6", mode='standard')
    scheduling_runs (df, "", mode='greedy')
    """

    df["cpu_start_time"] = 0
    df["cpu_finish_time"] = 0
    df["gpu_start_time"] = 0
    df["gpu_finish_time"] = 0
    df["cpu_start_cnt"] = 0
    df["cpu_finish_cnt"] = 0
    df["gpu_start_cnt"] = 0
    df["gpu_finish_cnt"] = 0

    # Initialization for cpu runs
    cpu_pre_run = df
    cpu_pre_run["processing_unit"] = "cpu"
    cpu_running = pd.DataFrame(columns=cpu_pre_run.columns)
    cpu_finish = pd.DataFrame(columns=cpu_pre_run.columns)

    # Initialization for gpu runs
    gpu_running = pd.DataFrame(columns=cpu_pre_run.columns)
    gpu_finish = pd.DataFrame(columns=gpu_running.columns)

    # max number of cpu threads running
    # max_thread = 8
    # max_thread = 16
    max_thread = 6

    if debug:
        print("After initialization")
        print("------------------------------------------------------")

    # Initialize for profiling record
    loop_counter = 0
    active_processes = []

    while_loop_start_time = time.time()

    # ####################################
    # Main loop
    # ####################################
    while (not cpu_finish.empty) | (gpu_finish.shape[0] != df.shape[0]):
        finished_processes = [
            proc for proc in active_processes if proc.poll() is not None
        ]
        for proc in finished_processes:
            proc.wait()  # Ensure the process resources are cleaned up
            active_processes.remove(proc)

        """
        Keep running until (1) all cpu jobs are finished and (2) all gpu jobs are finished 
        """

        if debug:
            print("------------------------------------------------------")
            print("Executing while loop to run all requested jobs")
            print()

        loop_counter += 1

        # ####################################
        # cpu_pre_run to cpu_running
        #
        # Allocate cpu jobs
        # ####################################
        while (not cpu_pre_run.empty) & (len(cpu_running) < max_thread):
            """
            This will run cpu processes up until (1) max #jobs runnings right now OR (2) no more cpu jobs to allocate
            """
            if debug:
                print("------------------------------------------------------")
                print("Allocating cpu jobs")

            for i in range(min(max_thread - len(cpu_running), len(cpu_pre_run))):
                # print (max_thread, len(cpu_running), i)
                # print(cpu_pre_run.iloc[0][['task_length', 'method', 'model_seed']])

                # Profiling
                if profiling:
                    cpu_start_time = time.time()
                    cpu_pre_run.at[0, "cpu_start_time"] = cpu_start_time
                    cpu_pre_run.at[0, "cpu_start_cnt"] = loop_counter
                    # print (cpu_pre_run.iloc[0])

                # Run exp
                process = run_experiment(cpu_pre_run.iloc[0], None)
                active_processes.append(process)

                # Attach to cpu_running
                cpu_running = pd.concat(
                    [cpu_running, cpu_pre_run.iloc[0].to_frame().T], ignore_index=True
                )
                # print (f'cpu_running.shape is {cpu_running.shape}')

                # Remove colum from pre_allocation
                cpu_pre_run = cpu_pre_run.drop([0]).reset_index(drop=True)
                # print(cpu_pre_run.shape)
                if debug:
                    print("***********************************")
                    print("move to cpu running")
                    print("***********************************")

            if debug:
                print("Finish allocating cpu jobs")
                # print (cpu_pre_run)
                # print (cpu_running)
                print("------------------------------------------------------")

        # ####################################
        # cpu_running to cpu_finish
        #
        # Verify if any cpu_running finishes running
        # ####################################
        if debug:
            print("------------------------------------------------------")
            print("Verifying if cpu_running finishes")
        rm_lst = []
        # print (cpu_running)
        for i in range(len(cpu_running)):
            """
            Check if current cpu_running job finishes running,
            if yes, move to cpu_finish and update cpu_running
            """
            # print (cpu_running)
            if not is_process_running(cpu_running.iloc[i]):
                # Profiling
                if profiling:
                    cpu_finish_time = time.time()
                    cpu_running.at[i, "cpu_finish_time"] = cpu_finish_time
                    cpu_running.at[i, "cpu_finish_cnt"] = loop_counter

                # Attach to cpu_finish
                cpu_finish = pd.concat(
                    [cpu_finish, cpu_running.iloc[i].to_frame().T], ignore_index=True
                )
                rm_lst.append(i)

                if debug:
                    print("***********************************")
                    print("move to cpu finish")
                    print("***********************************")

        if debug:
            # print (cpu_finish.head(10))
            print("Finish verifying if cpu jobs finish running")
            print("------------------------------------------------------")
        # Update cpu_running
        cpu_running = cpu_running.drop(rm_lst).reset_index(drop=True)

        # ####################################
        # cpu_finish to gpu_running
        #
        # Iterate through cpu_finish and try to allocate resources to run on gpu
        # ####################################
        if debug:
            print("------------------------------------------------------")
            print("Allocating cpu_finish to gpu_running")
        rm_lst = []
        cpu_finish["processing_unit"] = "gpu"
        for i in range(len(cpu_finish)):
            # Retrieve available gpu
            gpu_avai_id = next_avai_gpu(gpu_alloc, mode)

            print(i)

            if not gpu_avai_id == None:
                # Profiling
                if profiling:
                    gpu_start_time = time.time()
                    cpu_finish.at[i, "gpu_start_time"] = gpu_start_time
                    cpu_finish.at[i, "gpu_start_cnt"] = loop_counter

                # Attach to gpu_running
                gpu_running = pd.concat(
                    [gpu_running, cpu_finish.iloc[i].to_frame().T], ignore_index=True
                )
                gpu_running["processing_unit"] = "gpu"

                # Run exp with specified gpu_id

                process = run_experiment(cpu_finish.iloc[i], gpu_avai_id)
                active_processes.append(process)

                # Wait for 1 sec to let data transfer in previous round
                time.sleep(10)

                rm_lst.append(i)

                if debug:
                    print("***********************************")
                    print("move to gpu running")
                    print("***********************************")

        if debug:
            print("Finish allocating cpu_finish to gpu_running")
            print("------------------------------------------------------")

        # Update cpu_finish
        cpu_finish = cpu_finish.drop(rm_lst).reset_index(drop=True)

        # ####################################
        # gpu_running to gpu_finish
        #
        # Verify if any gpu_running finishes running
        # ####################################
        if debug:
            print("------------------------------------------------------")
            print("Verifying if gpu_running finish running")
        rm_lst = []
        for i in range(len(gpu_running)):
            if not is_process_running(gpu_running.iloc[i]):
                # Profiling
                if profiling:
                    gpu_finish_time = time.time()
                    gpu_running.at[i, "gpu_finish_time"] = gpu_finish_time
                    gpu_running.at[i, "gpu_finish_cnt"] = loop_counter

                # Current process finish running
                # print (gpu_running.iloc [i][['task_length', 'method', 'model_seed']])
                # Attach to gpu_finish
                gpu_finish = pd.concat(
                    [gpu_finish, gpu_running.iloc[i].to_frame().T], ignore_index=True
                )

                if debug:
                    print("***********************************")
                    print("move to gpu finish")
                    print("***********************************")

                rm_lst.append(i)
        if debug:
            print("Finish verifying if gpu_running finish")
            print("------------------------------------------------------")
        # Update gpu_running
        gpu_running = gpu_running.drop(rm_lst).reset_index(drop=True)

        # ####################################
        # Sleep until next round
        # ####################################

        # print ("Next round...")

        # print(cpu_finish.shape)
        # print(gpu_finish.shape[0])

    # #########################
    # End of while loop
    # #########################
    for proc in active_processes:
        proc.wait()

    while_loop_finish_time = time.time()
    while_loop_total_time = while_loop_finish_time - while_loop_start_time
    # while_loop_one_iter_time = while_loop_total_time / loop_counter

    # Profiling
    if profiling:
        profiling_analyze(
            df=gpu_finish,
            gpu_alloc=gpu_alloc,
            filename="profiling",
            total_time=while_loop_total_time,
            num_iter=loop_counter,
        )


def profiling_analyze(df, gpu_alloc, filename="profiling", total_time=0, num_iter=0):
    """
    This function will analyze each run in one scheduled bundle of jobs.
    The profiling will be stored under logs/profiling/{iteration_ids}
    """
    print(" ")
    print(" ")
    print(" ")
    print(" ")
    print(" ")
    print(" ")
    print(" ")
    print(" ")

    # id_value = 'debug_1'

    itr_id = df.iloc[0]["iteration_ids"]
    logdir = f"./logs/profiling/{itr_id}"
    os.makedirs(logdir, exist_ok=True)
    logpath = f"{logdir}/{filename}.log"

    # Calculate the difference and create a new column
    df["cpu_time"] = df["cpu_finish_time"] - df["cpu_start_time"]
    df["gpu_time"] = df["gpu_finish_time"] - df["gpu_start_time"]
    df["job_wait_time"] = df["gpu_start_time"] - df["cpu_finish_time"]
    df["job_wait_cnt"] = df["gpu_start_cnt"] - df["cpu_finish_cnt"]
    # df['one_iter_time'] = one_iter_time

    # one_iter_time = round (one_iter_time, 1)
    method = df.iloc[0]["method"]
    task_length = df.iloc[0]["task_length"]

    with open(logpath, "w") as file:
        file.write(
            f"########################################################################################\n"
        )
        file.write(
            f"Profiling of itr_id={itr_id} scheduled runs. \nDetailed log see below. \n\n"
        )
        file.write(
            f"cnt:   means at which iteration of scheduler the job is scheduled. \n"
        )
        file.write(f"cpu_time:   time (seconds) to create dataset \n")
        file.write(f"gpu_time:   time (seconds) to run jobs on gpu \n")
        file.write(
            f"job_wait_time:   time the job is waiting to be assigned to run on gpu\n"
        )
        file.write(
            f"########################################################################################\n\n"
        )
        file.write(f"number_of_jobs={len (df)}\n\n")
        file.write(f"total_time={total_time}\n")
        file.write(f"one_iter_time={round (total_time / num_iter,1)}\n")
        file.write(f"loops={num_iter}\n")
        # file.write(f'method={method}\n')
        # file.write(f'task_length={task_length}\n')

    # df_group = df.groupby(['task_length', 'method'])[['cpu_start_cnt', 'cpu_finish_cnt', 'cpu_time_difference', 'cpu_start_cnt', 'cpu_finish_cnt', 'gpu_time_difference']].describe ()
    df_group = df.groupby(["task_length", "method"])[
        [
            "cpu_time",
            "gpu_time",
            "cpu_start_cnt",
            "job_wait_time",
            "job_wait_cnt",
            "cpu_finish_cnt",
            "gpu_start_cnt",
            "gpu_finish_cnt",
        ]
    ].agg(["mean", "median", "min", "max"])

    # Flatten the MultiIndex for columns
    df_group.columns = ["_".join(col).strip() for col in df_group.columns.values]

    # Reset index to make 'task_length' and 'method' regular columns
    df_group.reset_index(inplace=True)

    with open(logpath, "a") as f:
        for index, row in df_group.iterrows():
            f.write("\n")  # Start of a new group

            for col in df_group.columns:
                f.write(
                    f"{col}={row[col]}\n"
                )  # Adjusted for consistency with your desired output
                if (
                    ("cpu_time_max" in col)
                    | ("method" in col)
                    | ("cpu_start_cnt_max" in col)
                    | ("cpu_finish_cnt_max" in col)
                    | ("gpu_start_cnt_max" in col)
                    | ("gpu_finish_cnt_max" in col)
                    | ("gpu_time_max" in col)
                    | ("job_wait_cnt_max" in col)
                    | ("job_wait_time_max" in col)
                ):  # Check if the current column is the last CPU metric
                    f.write("\n")  # Insert a line break between CPU and GPU metrics

            f.write("\n")  # End of the group
        f.write("Detailed runs \n")

    # Export  to CSV
    df[
        [
            "task_length",
            "method",
            "cpu_start_cnt",
            "cpu_finish_cnt",
            "cpu_time",
            "gpu_start_cnt",
            "gpu_finish_cnt",
            "gpu_time",
            "job_wait_cnt",
            "job_wait_time",
        ]
    ].to_csv(logpath, mode="a", index=False)

    print("After profiling")
    print(df_group)
    print(" ")
    print(" ")
    print(" ")
    print(" ")
    print(" ")
    print(" ")
    print(" ")
    # print (df [['cpu_start_time', 'cpu_finish_time']])


def main(csv_filepath: str, gpu_alloc: str = "0 1 2 3 4 5 6 7", profiling: bool = True):
    df = pd.read_csv(csv_filepath)

    df["est_compute"] = df["est_compute"].astype(int)

    # # Include columns for profiling
    # df['cpu_start_time'] = 0
    # df['cpu_finish_time'] = 0
    # df['gpu_start_time'] = 0
    # df['gpu_finish_time'] = 0

    # profileing_analyze (df)

    # scheduling_runs (df, gpu_alloc, mode='standard', debug=True, profiling=True)


if __name__ == "__main__":
    typer.run(main)

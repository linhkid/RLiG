import os
import subprocess


def run_command(command):
    # command = "ping -c 4 127.0.0.1"
    print(f"Running command: {command}")
    try:
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        for line in iter(process.stdout.readline, ""):
            print(line, end="")  # 输出标准输出的每一行

        process.wait()

        if process.returncode != 0:
            print(f"Command failed with return code {process.returncode}")
            for stderr_line in iter(process.stderr.readline, ""):
                print(stderr_line, end="")
        else:
            print("Command finished successfully.")
    except Exception as e:
        print(f"Error occurred: {e}")


if __name__ == '__main__':
    # available_datasets = ["magic", "satellite", "loan", "connect", "credit", "localization-dm"]
    available_datasets = ["covtype", "pokerhand"]
    # Finished "adult", "chess", "letter", "pokerhand" "car", "nursery",
    # available_methods = ["gran", "notears", "dag_gnn"]
    available_methods = ["notears"]

    for dataset_name in available_datasets:
        for method in available_methods:
            # Define variables
            DATA_SET_NAME = dataset_name
            CODE_PATH = "./"  # path to the code root
            METHOD_NAME = method
            EXP_PATH = f"./results_{METHOD_NAME}/{DATA_SET_NAME}"  # folder to contain all artifacts saved by the program
            DATA_PATH = f"./uci_data/{DATA_SET_NAME}"  # path to data (data1.npy, DAG1.npy, CPDAG1.npy, ...)
            MODEL = "NonLinGaussANM"  # or NonLinGauss
            DATA_INDEX = 1  # Choose which dataset to use
            NUM_VARS = 10  # should match the data provided

            # Create the directory if it doesn't exist
            os.makedirs(EXP_PATH, exist_ok=True)

            print(f"start evaluating {method} on {dataset_name}")
            try:
                if method == "gran":
                    cmd_GraN_DAG = (
                        f"singularity exec --nv --containall "
                        f"-B {DATA_PATH}:/dataset/ -B {EXP_PATH}:/final_log/ -B {CODE_PATH}:/code/ ./container.simg "
                        f"bash -c \"cd /code && python -u main.py --exp-path /final_log/ --data-path /dataset/ "
                        f"--i-dataset {DATA_INDEX} --model {MODEL} --train --to-dag --num-vars {NUM_VARS} --jac-thresh\""
                    )
                    run_command(cmd_GraN_DAG)
                elif method == "notears":
                    cmd_notears = (
                        f"singularity exec --nv --containall "
                        f"-B {DATA_PATH}:/dataset/ -B {EXP_PATH}:/final_log/ -B {CODE_PATH}:/code/ ./container.simg "
                        f"bash -c \"cd /code && python -u notears/main.py --exp-path /final_log/ --data-path /dataset/ "
                        f"--i-dataset {DATA_INDEX}\""
                    )
                    run_command(cmd_notears)
                elif method == "dag_gnn":
                    cmd_dag_gnn = (
                        f"singularity exec --nv --containall "
                        f"-B {DATA_PATH}:/dataset/ -B {EXP_PATH}:/final_log/ -B {CODE_PATH}:/code/ ./container.simg "
                        f"bash -c \"cd /code && python -u dag_gnn/main.py --exp-path /final_log/ --data-path /dataset/ "
                        f"--i-dataset {DATA_INDEX}\""
                    )
                    run_command(cmd_dag_gnn)
                else:
                    raise "Unmatched SOTA"
            except Exception as e:
                print(e)
                print(f"Error occurs in {dataset_name} using {method}")
                raise Exception
            except KeyboardInterrupt:
                print(f"Stopped at {dataset_name} using {method}")
                raise KeyboardInterrupt

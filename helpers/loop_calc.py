#!/usr/bin/env python3
import math
import readchar
import json
import os

ELT_SIZE = 4
CONFIG_FILE = 'gpu_configurations.json'
MAX_THREAD_BLOCK_SIZE = 1024

def load_configurations():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as file:
            return json.load(file)
    return {}

def save_configurations(configurations):
    with open(CONFIG_FILE, 'w') as file:
        json.dump(configurations, file, indent=4)

def calculate_tb_size(l1_total, concurrent_tb):
    total_elts_l1 = math.floor(l1_total / ELT_SIZE)
    tb_size = math.floor(total_elts_l1 / concurrent_tb)
    tb_size = min(tb_size, MAX_THREAD_BLOCK_SIZE)
    return tb_size

def calculate_global_tb_size(global_memory_capacity, concurrent_tb):
    total_elts = math.floor(global_memory_capacity / ELT_SIZE)
    tb_size = math.floor(total_elts / concurrent_tb)
    tb_size = min(tb_size, MAX_THREAD_BLOCK_SIZE)
    return tb_size

def calculate_max_tb_size():
    return MAX_THREAD_BLOCK_SIZE

def calculate_global_loops(global_memory_capacity, concurrent_tb, tb_size):
    total_elts = math.floor(global_memory_capacity / ELT_SIZE)
    global_loops = math.floor(total_elts / (concurrent_tb * tb_size))
    return global_loops

def calculate_l1_loops(l1_total, concurrent_tb, tb_size):
    l1_loops = math.floor(math.floor(l1_total / ELT_SIZE) / (concurrent_tb * tb_size))
    return l1_loops

def display_gpu_info(name, config):
    print(f"\nUsing predefined values for {name}:")
    for key, value in config.items():
        print(f"{key.replace('_', ' ').title()}: {value}")

    l1_tb_size = calculate_tb_size(config["l1_total"], config["concurrent_tb"])
    global_tb_size = calculate_global_tb_size(config["global_memory_capacity"], config["concurrent_tb"])
    max_tb_size = config.get("max_tb_size", MAX_THREAD_BLOCK_SIZE)

    global_loops = calculate_global_loops(config["global_memory_capacity"], config["concurrent_tb"], global_tb_size)
    print(f"Global loops: {global_loops}")

    l1_loops = calculate_l1_loops(config["l1_total"], config["concurrent_tb"], l1_tb_size)
    print(f"L1 loops: {l1_loops}")

    print(f"L1 thread block size: {l1_tb_size}")
    print(f"Global thread block size: {global_tb_size}")
    print(f"Max thread block size: {max_tb_size}")

def print_menu(gpus, selected_index):
    print("\033c", end="")  # Clear the console
    print("Select an option using arrow keys and press Enter:")
    for i, gpu in enumerate(gpus):
        if i == selected_index:
            print(f"> {gpu}")
        else:
            print(f"  {gpu}")

def select_gpu(gpus):
    gpus_with_options = gpus + ["new", "delete", "edit", "calculate", "quit"]
    index = 0
    while True:
        print_menu(gpus_with_options, index)

        key = readchar.readkey()

        if (key == readchar.key.UP or key == readchar.key.LEFT) and index > 0:
            index -= 1
        elif (key == readchar.key.DOWN or key == readchar.key.RIGHT) and index < len(gpus_with_options) - 1:
            index += 1
        elif key == readchar.key.ENTER:
            return gpus_with_options[index]

def delete_gpu(gpus):
    index = 0
    while True:
        print_menu(gpus, index)
        print("Select a GPU to delete using arrow keys and press Enter:")

        key = readchar.readkey()

        if (key == readchar.key.UP or key == readchar.key.LEFT) and index > 0:
            index -= 1
        elif (key == readchar.key.DOWN or key == readchar.key.RIGHT) and index < len(gpus) - 1:
            index += 1
        elif key == readchar.key.ENTER:
            return gpus[index]

def edit_gpu(gpu_configurations):
    gpus = list(gpu_configurations.keys())
    gpu_to_edit = select_gpu(gpus)
    if gpu_to_edit in gpu_configurations:
        print(f"\nEditing configuration for {gpu_to_edit}:")
        for key, value in gpu_configurations[gpu_to_edit].items():
            new_value = input(f"Enter new value for {key.replace('_', ' ').title()} (current: {value}): ")
            if new_value:
                try:
                    if key in ["global_memory_capacity", "l1_total", "constant", "shared_memory_per_block", "registers_per_block", "l1_per_sm", "l2"]:
                        gpu_configurations[gpu_to_edit][key] = int(new_value)
                    else:
                        gpu_configurations[gpu_to_edit][key] = float(new_value)
                except ValueError:
                    print(f"Invalid input for {key}. Keeping current value.")

        # Recalculate thread block sizes and max_tb_size
        config = gpu_configurations[gpu_to_edit]
        config["l1_tb_size"] = calculate_tb_size(config["l1_total"], config["concurrent_tb"])
        config["global_tb_size"] = calculate_global_tb_size(config["global_memory_capacity"], config["concurrent_tb"])
        config["max_tb_size"] = calculate_max_tb_size()

        # Save updated configurations to file
        save_configurations(gpu_configurations)
    else:
        print("GPU configuration not found.")

def handle_add_new_gpu(gpu_configurations):
    gpu_name = input("Enter the GPU name: ").lower()
    global_memory_capacity = int(input("Enter the global memory capacity (in bytes): "))
    concurrent_tb = int(input("Enter the number of concurrent thread blocks: "))
    sm = int(input("Enter the number of streaming multiprocessors (SM): "))
    constant = int(input("Enter the size of the constant memory (in bytes): "))
    shared_memory_per_block = int(input("Enter the shared memory per block (in bytes): "))
    registers_per_block = int(input("Enter the number of registers per block: "))
    threads_per_sm = int(input("Enter the number of threads per SM: "))
    l1_per_sm = int(input("Enter the size of L1 cache per SM (in bytes): "))
    l1_total = int(input("Enter the total L1 cache size (in bytes): "))
    l2 = int(input("Enter the size of L2 cache (in bytes): "))

    # Calculate thread block sizes
    l1_tb_size = calculate_tb_size(l1_total, concurrent_tb)
    global_tb_size = calculate_global_tb_size(global_memory_capacity, concurrent_tb)
    max_tb_size = calculate_max_tb_size()

    # Ensure calculated sizes do not exceed constraints
    l1_tb_size = min(l1_tb_size, max_tb_size)
    global_tb_size = min(global_tb_size, max_tb_size)

    # Add new GPU configuration
    gpu_configurations[gpu_name] = {
        "global_memory_capacity": global_memory_capacity,
        "concurrent_tb": concurrent_tb,
        "sm": sm,
        "constant": constant,
        "shared_memory_per_block": shared_memory_per_block,
        "registers_per_block": registers_per_block,
        "threads_per_sm": threads_per_sm,
        "l1_per_sm": l1_per_sm,
        "l1_total": l1_total,
        "l2": l2,
        "l1_tb_size": l1_tb_size,
        "global_tb_size": global_tb_size,
        "max_tb_size": max_tb_size
    }

    # Save updated configurations to file
    save_configurations(gpu_configurations)

def handle_delete_gpu(gpu_configurations):
    gpus = list(gpu_configurations.keys())
    gpu_to_delete = delete_gpu(gpus)
    if gpu_to_delete in gpu_configurations:
        del gpu_configurations[gpu_to_delete]
        print(f"Deleted GPU configuration for: {gpu_to_delete}")
        # Save updated configurations to file
        save_configurations(gpu_configurations)
    else:
        print("No GPU configuration deleted.")

def handle_calculate(gpu_configurations):
    gpus = list(gpu_configurations.keys())
    gpu_to_calculate = select_gpu(gpus)
    if gpu_to_calculate in gpu_configurations:
        config = gpu_configurations[gpu_to_calculate]
        l1_tb_size = calculate_tb_size(config["l1_total"], config["concurrent_tb"])
        global_tb_size = calculate_global_tb_size(config["global_memory_capacity"], config["concurrent_tb"])

        global_loops = calculate_global_loops(config["global_memory_capacity"], config["concurrent_tb"], global_tb_size)
        l1_loops = calculate_l1_loops(config["l1_total"], config["concurrent_tb"], l1_tb_size)

        print(f"\nRecalculated values for {gpu_to_calculate}:")
        print(f"L1 thread block size: {l1_tb_size}")
        print(f"Global thread block size: {global_tb_size}")
        print(f"Max thread block size: {config.get('max_tb_size', MAX_THREAD_BLOCK_SIZE)}")
        print(f"Global loops: {global_loops}")
        print(f"L1 loops: {l1_loops}")
    else:
        print("GPU configuration not found.")

def main():
    gpu_configurations = load_configurations()
    gpus = list(gpu_configurations.keys())

    while True:
        selected_option = select_gpu(gpus)

        if selected_option in gpu_configurations:
            display_gpu_info(selected_option, gpu_configurations[selected_option])
            input("Press Enter to continue...")

        elif selected_option == "new":
            handle_add_new_gpu(gpu_configurations)
            gpus = list(gpu_configurations.keys())  # Refresh GPU list after adding

        elif selected_option == "delete":
            handle_delete_gpu(gpu_configurations)
            gpus = list(gpu_configurations.keys())  # Refresh GPU list after deleting

        elif selected_option == "edit":
            edit_gpu(gpu_configurations)
            gpus = list(gpu_configurations.keys())  # Refresh GPU list after editing

        elif selected_option == "calculate":
            handle_calculate(gpu_configurations)

        elif selected_option == "quit":
            print("Exiting program.")
            break

        else:
            print("Invalid selection. Please try again.")
            input("Press Enter to continue...")

if __name__ == "__main__":
    main()


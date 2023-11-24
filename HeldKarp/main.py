import time
import os
import csv
import psutil
from tqdm import tqdm
from itertools import combinations


def held_karp_algorithm(distance_matrix):
    n = len(distance_matrix)
    C = {}

    for k in range(1, n):
        C[(1 << k, k)] = (distance_matrix[0][k], 0)

    for subset_size in range(2, n):
        for subset in combinations(range(1, n), subset_size):
            bits = 0
            for bit in subset:
                bits |= 1 << bit
            for k in subset:
                prev = bits & ~(1 << k)
                res = []
                for m in subset:
                    if m == 0 or m == k:
                        continue
                    res.append((C[(prev, m)][0] + distance_matrix[m][k], m))
                C[(bits, k)] = min(res)

    bits = (2**n - 1) - 1
    res = []
    for k in range(1, n):
        res.append((C[(bits, k)][0] + distance_matrix[k][0], k))
    opt, parent = min(res)
    path = []
    for i in range(n - 1):
        path.append(parent)
        new_bits = bits & ~(1 << parent)
        _, parent = C[(bits, parent)]
        bits = new_bits
    path.append(0)

    return opt, list(reversed(path))


def read_tsp_data(filename):
    full_path = os.path.join("data", filename)
    with open(full_path, "r") as file:
        lines = file.read().splitlines()
        size = int(lines[0])
        matrix = []
        for line in lines[1:]:
            matrix.append(list(map(int, line.split())))
        return matrix


def parse_config(config_file):
    tsp_instances = []
    with open(config_file, "r") as file:
        for line in file:
            parts = line.split()
            instance = {
                "file_name": parts[0],
                "times_to_run": int(parts[1]),
                "number_of_instances": int(parts[2]),
            }
            tsp_instances.append(instance)
    return tsp_instances


def main():
    # Main function to run the program
    config_file = "config.ini"
    tsp_instances = parse_config(config_file)

    # Specify the correct delimiter here, if your system expects something other than a comma
    delimiter = ","

    # Open the CSV file in write mode
    with open("tsp_hk_results.csv", mode="w", newline="") as file:
        writer = csv.writer(
            file, delimiter=delimiter, quotechar='"', quoting=csv.QUOTE_MINIMAL
        )

        # Write the header if the file is new/empty
        writer.writerow(["instance", "time", "cost", "path", "peak_memory_usage_in_MB"])

        for instance in tsp_instances:
            distance_matrix = read_tsp_data(instance["file_name"])
            for _ in tqdm(
                range(instance["times_to_run"]),
                desc=f"Processing {instance['file_name']}",
            ):
                start_time = time.time_ns()
                # gc.disable()
                initial_memory = psutil.Process().memory_info().rss

                cost, path = held_karp_algorithm(distance_matrix)

                peak_memory = psutil.Process().memory_info().rss
                end_time = time.time_ns()
                # gc.enable()
                # Calculate execution time in microseconds
                execution_time_ns = end_time - start_time

                # Calculate peak memory usage in MB
                peak_memory_usage_mb = (peak_memory - initial_memory) / 1024 / 1024

                # Format the path as a string if necessary
                path_str = "-".join(map(str, path))

                # Write the result immediately to the CSV file
                writer.writerow(
                    [
                        instance["file_name"],
                        execution_time_ns,
                        cost,
                        path_str,
                        peak_memory_usage_mb,
                    ]
                )
    input("Press anything to exit")


if __name__ == "__main__":
    main()

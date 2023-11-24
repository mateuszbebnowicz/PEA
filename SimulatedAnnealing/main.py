import math
import random
import csv
from tqdm import tqdm
import os
import time


def geometric_cooling(current_temp, iteration, cooling_rate):
    # Ensure the cooling rate is between 0 and 1.
    assert 0 < cooling_rate < 1, "Cooling rate must be between 0 and 1."
    return max(current_temp * cooling_rate, 1e-10)  # Prevent temperature from reaching absolute zero


def boltzmann_cooling(current_temp, iteration, k):
    # Ensure the constant k is positive.
    assert k > 0, "Constant k must be positive."
    return max(current_temp / math.log(iteration + k), 1e-10)  # Prevent temperature from reaching absolute zero


# Define initial temperature strategie
def fixed_temperature(value):
    return value


# Define solution choice strategies
def greedy_initial_solution(distance_matrix):
    # Starts at the first city and always goes to the nearest city not yet visited
    n = len(distance_matrix)
    start = 0  # Can be randomized or selected by some criteria
    unvisited = set(range(n))
    unvisited.remove(start)
    solution = [start]

    current_city = start
    while unvisited:
        next_city = min(unvisited, key=lambda city: distance_matrix[current_city][city])
        unvisited.remove(next_city)
        solution.append(next_city)
        current_city = next_city

    return solution


# Define swap strategies
def two_opt_swap(solution):
    # Randomly pick two edges and perform a 2-opt swap
    city1 = random.randint(1, len(solution) - 2)
    city2 = random.randint(city1 + 1, len(solution) - 1)
    new_solution = solution[:city1] + solution[city1:city2][::-1] + solution[city2:]
    return new_solution


def subset_shuffle_swap(solution):
    # Randomly choose two indices to determine the subset to shuffle
    start = random.randint(0, len(solution) - 2)
    end = random.randint(start + 1, len(solution) - 1)

    # Extract the subset and shuffle it
    subset = solution[start:end]
    random.shuffle(subset)

    # Replace the original subset with the shuffled subset
    new_solution = solution[:start] + subset + solution[end:]

    return new_solution


# SA implementation
def simulated_annealing(distance_matrix, initial_temp, cooling_schedule, swap_strategy, initial_solution, epoch_length, cooling_schedule_args):
    current_solution = initial_solution
    current_temp = initial_temp
    best_solution = current_solution
    best_cost = calculate_cost(distance_matrix, best_solution)
    iteration = 0

    while not stopping_condition(current_temp, iteration):
        for _ in range(epoch_length):  # Iterate for epoch_length times before cooling
            new_solution = swap_strategy(current_solution)
            new_cost = calculate_cost(distance_matrix, new_solution)
            cost_difference = new_cost - best_cost

            if cost_difference < 0 or math.exp(-cost_difference / current_temp) > random.random():
                current_solution = new_solution
                if new_cost < best_cost:
                    best_solution = new_solution
                    best_cost = new_cost

            # Update the current temperature using the selected cooling schedule
            current_temp = cooling_schedule(current_temp, iteration, **cooling_schedule_args)
            iteration += 1

    return best_cost, best_solution


def calculate_cost(distance_matrix, path):
    cost = 0
    for i in range(1, len(path)):
        if path[i - 1] >= len(distance_matrix) or path[i] >= len(distance_matrix):
            raise IndexError(
                f"City index out of range: {path[i-1]} or {path[i]} not in distance matrix."
            )
        if path[i - 1] < 0 or path[i] < 0:
            raise IndexError(
                f"City index cannot be negative: {path[i-1]} or {path[i]}."
            )
        cost += distance_matrix[path[i - 1]][path[i]]
    if path:
        cost += distance_matrix[path[-1]][path[0]]
    return cost


# Define stopping condition
def stopping_condition(current_temp, iteration):
    # Define a minimum temperature threshold to avoid division by zero
    MIN_TEMPERATURE = 1e-10
    MAX_ITERATIONS = 100000
    return current_temp < MIN_TEMPERATURE or iteration > MAX_ITERATIONS


# Define a dictionary to map strategy names to functions
cooling_schedules = {
    "geometric_cooling": geometric_cooling,
    "boltzmann_cooling": boltzmann_cooling,
}

swap_strategies = {
    "two_opt_swap": two_opt_swap,
    "subset_shuffle_swap": subset_shuffle_swap,
}

initial_temperature_strategies = {
    "fixed_temperature": fixed_temperature,
}

initial_solution_strategies = {
    "greedy_initial_solution": greedy_initial_solution,
}


def parse_config(config_file):
    tsp_instances = []
    settings = {
        "cooling_temperature": None,
        "cooling_schedule": None,
        "swap_strategy": None,
        "initial_temperature_strategy": None,
        "initial_solution_strategy": None,
        "cooling_rate": 0.95,
        "k": 1,
    }
    with open(config_file, "r") as file:
        lines = file.readlines()
        for setting_line in lines:
            if "=" in setting_line:
                key, value = setting_line.strip().split("=")
                if key in settings:
                    settings[key] = float(value.strip()) if key in ["cooling_rate", "k"] else value.strip()
            else:
                parts = setting_line.split()
                if len(parts) == 3:
                    instance = {
                        "file_name": parts[0],
                        "times_to_run": int(parts[1]),
                        "number_of_instances": int(parts[2]),
                    }
                    tsp_instances.append(instance)
    return settings, tsp_instances


def read_tsp_data(filename):
    full_path = os.path.join("data", filename)
    with open(full_path, "r") as file:
        lines = file.read().splitlines()
        size = int(lines[0])
        matrix = []
        for line in lines[1:size + 1]:
            matrix.append(list(map(int, line.split())))
        best_known_solution = int(lines[size + 1])
        return matrix, best_known_solution


# Main function to run the program
def main():
    config_file = 'config.ini'
    settings, tsp_instances = parse_config(config_file)

    initial_temp_strategy = initial_temperature_strategies[settings['initial_temperature_strategy']]
    initial_temp = initial_temp_strategy(float(settings['cooling_temperature']))

    initial_solution_strategy = initial_solution_strategies[settings['initial_solution_strategy']]

    selected_cooling_schedule = cooling_schedules[settings["cooling_schedule"]]
    selected_swap_strategy = swap_strategies[settings["swap_strategy"]]

    # Define a range of epoch lengths to test
    epoch_lengths = [10, 50, 100, 500, 1000]

    # Specify the correct delimiter here
    delimiter = ";"

    # Open the CSV file in write mode
    with open("tsp_sa_results.csv", mode="w", newline="") as file:
        writer = csv.writer(
            file, delimiter=delimiter, quotechar='"', quoting=csv.QUOTE_MINIMAL
        )

        # Write the header if the file is new/empty
        writer.writerow(["instance", "time", "cost", "path", "error in %"])

        # Load or define your distance matrix
        for epoch_length in epoch_lengths:
            writer.writerow([epoch_length])
            for instance in tsp_instances:
                distance_matrix, best_known_solution = read_tsp_data(instance["file_name"])
                initial_solution = initial_solution_strategy(distance_matrix)

                # Set up the parameters for simulated annealing
                cooling_schedule_args = {}
                if settings["cooling_schedule"] == "geometric_cooling":
                    cooling_schedule_args['cooling_rate'] = settings['cooling_rate']
                elif settings["cooling_schedule"] == "boltzmann_cooling":
                    cooling_schedule_args['k'] = settings['k']

                for _ in tqdm(
                    range(instance["times_to_run"]),
                    desc=f"Processing {instance['file_name']}",
                ):
                    # Start SA
                    start_time = time.time_ns()

                    cost, path = simulated_annealing(
                        distance_matrix=distance_matrix,
                        initial_temp=initial_temp,
                        cooling_schedule=selected_cooling_schedule,
                        swap_strategy=selected_swap_strategy,
                        initial_solution=initial_solution,
                        epoch_length=epoch_length,
                        cooling_schedule_args=cooling_schedule_args
                    )

                    end_time = time.time_ns()

                    # Calculate error size in %
                    error_size_percentage = abs((cost - best_known_solution)/cost) * 100

                    # Calculate execution time in microseconds
                    execution_time_ns = end_time - start_time

                    # Format the path as a string
                    path_str = "-".join(map(str, path))

                    # Write the result immediately to the CSV file
                    writer.writerow(
                        [
                            instance["file_name"],
                            execution_time_ns,
                            cost,
                            path_str,
                            error_size_percentage
                        ]
                    )
    input("Press anything to exit")


if __name__ == "__main__":
    main()

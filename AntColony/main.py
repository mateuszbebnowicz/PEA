import csv
import os
import time
import numpy as np
from tqdm import tqdm


class AntColonyOptimization:
    def __init__(self, cities, alpha, beta, rho, m, tau0):
        self.cities = cities
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.m = m
        self.tau0 = tau0
        self.pheromone = np.full((len(cities), len(cities)), tau0)
        self.visibility = 1 / self.calculate_distances()

    def run(self):
        best_path = None
        best_distance = float("inf")

        for _ in range(self.m):
            path, distance = self.construct_solution()
            if distance < best_distance:
                best_path, best_distance = path, distance
            self.update_pheromones(path, distance)

        return best_path, best_distance

    def calculate_distances(self):
        num_cities = len(self.cities)
        distances = np.zeros((num_cities, num_cities))
        for i in range(num_cities):
            for j in range(num_cities):
                if i != j:
                    distances[i][j] = self.cities[i][j]
                else:
                    distances[i][j] = np.inf
        return distances

    def construct_solution(self):
        num_cities = len(self.cities)
        path = [0]  # Start from a random city
        while len(path) < num_cities:
            current_city = path[-1]
            probabilities = self.calculate_transition_probabilities(current_city, path)
            next_city = self.choose_next_city(probabilities)
            path.append(next_city)
        return path, self.calculate_path_distance(path)

    def calculate_transition_probabilities(self, current_city, path):
        probabilities = []
        for city in range(len(self.cities)):
            if city not in path:
                pheromone = self.pheromone[current_city][city] ** self.alpha
                visibility = self.visibility[current_city][city] ** self.beta
                probabilities.append(pheromone * visibility)
            else:
                probabilities.append(0)
        probabilities_sum = sum(probabilities)
        if probabilities_sum == 0:
            return [1.0 / len(self.cities)] * len(self.cities)  # Equal probability if all are zero
        probabilities = [p / probabilities_sum for p in probabilities]
        return probabilities

    def choose_next_city(self, probabilities):
        return np.random.choice(len(probabilities), p=probabilities)

    def calculate_path_distance(self, path):
        distance = 0
        for i in range(len(path) - 1):
            distance += self.cities[path[i]][path[i + 1]]
        distance += self.cities[path[-1]][path[0]]  # Return to the starting city
        return distance

    def update_pheromones(self, path, distance):
        # Evaporate some of the existing pheromone
        self.pheromone *= 1 - self.rho

        # Add new pheromone
        pheromone_to_add = 1 / distance
        for i in range(len(path) - 1):
            self.pheromone[path[i]][path[i + 1]] += pheromone_to_add
            self.pheromone[path[i + 1]][
                path[i]
            ] += pheromone_to_add


def parse_config(config_file):
    tsp_instances = []
    settings = {
        "alpha": None,
        "beta": None,
        "rho": None,
    }
    with open(config_file, "r") as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip()
            if "=" in line:
                key, value = line.split("=")
                key = key.strip()
                value = value.split(";")[0].strip()
                if key in settings:
                    settings[key] = float(value)
            elif line:
                parts = line.split()
                if len(parts) >= 2:
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
        for line in lines[1: size + 1]:
            matrix.append(list(map(int, line.split())))
        best_known_solution = int(
            lines[size + 1]
        )
        return matrix, best_known_solution


delimiter = ";"

if __name__ == "__main__":
    params, tsp_instances = parse_config("config.ini")
    with open("tsp_aco_results.csv", mode="w", newline="") as file:
        writer = csv.writer(
            file, delimiter=delimiter, quotechar='"', quoting=csv.QUOTE_MINIMAL
        )

        # Write the header if the file is new/empty
        writer.writerow(["time", "cost", "path", "error in %"])

        for instance in tsp_instances:
            distance_matrix, best_known_solution = read_tsp_data(instance["file_name"])

            writer.writerow([instance["file_name"], best_known_solution])
            for _ in tqdm(
                    range(instance["times_to_run"]),
                    desc=f"Processing {instance['file_name']}",
            ):
                aco = AntColonyOptimization(
                    distance_matrix,
                    alpha=float(params["alpha"]),
                    beta=float(params["beta"]),
                    rho=float(params["rho"]),
                    m=int(instance["number_of_instances"]),
                    tau0=float(instance["number_of_instances"] / best_known_solution),
                )
                start_time = time.time_ns()

                best_path, best_distance = aco.run()

                end_time = time.time_ns()

                # Calculate error size in %
                error_size_percentage = abs((best_distance - best_known_solution)/best_known_solution) * 100

                # Calculate execution time in microseconds
                execution_time_ns = end_time - start_time

                # Format the path as a string if necessary
                path_str = "-".join(map(str, best_path))

                # Write the result immediately to the CSV file
                writer.writerow(
                    [
                        execution_time_ns,
                        best_distance,
                        path_str,
                        error_size_percentage
                    ]
                )

import numpy as np
import random
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import argparse
import multiprocessing as mp
import logging


'''Change the names of the these loaded files to be your functional connectome numpy files.
In my runs I use a (43,116,116) numpy. 43 is the number of individuals and 116x116 is the regions of the functional connectome.
You can use any number of individuals or any type of atlas you want, so long as you change initial region configuration and generate neighbor function
to the range of indices of your desired atlas.
'''
session1Segment1 = np.load("session1Segment1.npy")
session2Segment1 = np.load("session2Segment1.npy")
session1Segment2 = np.load("session1Segment2.npy")
session2Segment2 = np.load("session2Segment2.npy")


'''Extracting regions from functional connectome (FC): this picks regions from the FC which are selected by the optimization.
'''
def getCorr(individual, data):
    individual_array = np.array(individual)
    selected_elements = []
    for i in range(len(individual_array)):
        for j in range(i + 1, len(individual_array)):
            selected_elements.append(data[individual_array[i], individual_array[j]])
    return np.array(selected_elements)

def hamming_distance(list1, list2):
    if len(list1) != len(list2):
        raise ValueError("Lists must be of the same length")
    return sum(el1 != el2 for el1, el2 in zip(list1, list2))

'''Evaluating solution: this decides the fitness of the sub-network. This requires both segments to have good fingerprinting accuracy.
The equation is available on the readme.
'''
def evaluate_solution(individual, session1Segment1, session2Segment1, session1Segment2, session2Segment2):
        finalFitness = 0
        listsOfContrastingAccuracy = []
        listsOfCorrectAccuracy = []
        listsOfContrastingAccuracy2 = []
        listsOfCorrectAccuracy2 = []
        fitnessClassifications = []
        fitnessClassificationsOtherSegment = []
        for i in range(len(session1Segment1)):
            listsOfContrastingAccuracy = []
            filtered_arr = getCorr(individual, session1Segment1[i])
            otherSession = session2Segment1[i]
            filtered_arr2 = getCorr(individual, otherSession)
            identifCorr = abs(round(np.corrcoef(filtered_arr2.flatten(), filtered_arr.flatten())[1][0],5))
            listsOfCorrectAccuracy.append(identifCorr)
            for j in range(len(session1Segment1)):
                if j != i:
                  filteredArr3 = getCorr(individual, session1Segment1[j])
                  filteredArr4 = getCorr(individual, session2Segment1[j])
                  identifCorr2 = abs(round(np.corrcoef(filteredArr3.flatten(), filtered_arr.flatten())[1][0],5))
                  identifCorr3 = abs(round(np.corrcoef(filteredArr4.flatten(), filtered_arr.flatten())[1][0],5))
                  listsOfContrastingAccuracy.append(identifCorr2)
                  listsOfContrastingAccuracy.append(identifCorr3)

            if identifCorr > max(listsOfContrastingAccuracy):
                fitnessClassifications.append(1)
            else:
                fitnessClassifications.append(0)
            
        for i in range(len(session1Segment2)):
            listsOfContrastingAccuracy2 = []
            filtered_arr = getCorr(individual, session1Segment2[i])
            otherSession = session2Segment2[i]
            filtered_arr2 = getCorr(individual, otherSession)
            identifCorr2 = abs(round(np.corrcoef(filtered_arr2.flatten(), filtered_arr.flatten())[1][0],5))
            listsOfCorrectAccuracy2.append(identifCorr)
            for j in range(len(session1Segment1)):
                if j != i:
                  filteredArr3 = getCorr(individual, session1Segment2[j])
                  filteredArr4 = getCorr(individual, session2Segment2[j])
                  identifCorr4 = abs(round(np.corrcoef(filteredArr3.flatten(), filtered_arr.flatten())[1][0],5))
                  identifCorr3 = abs(round(np.corrcoef(filteredArr4.flatten(), filtered_arr.flatten())[1][0],5))
                  listsOfContrastingAccuracy2.append(identifCorr4)
                  listsOfContrastingAccuracy2.append(identifCorr3)

            if identifCorr2 > max(listsOfContrastingAccuracy2):
                fitnessClassificationsOtherSegment.append(1)
            else:
                fitnessClassificationsOtherSegment.append(0)

        finalFitness = (min((np.sum(fitnessClassificationsOtherSegment)),(np.sum(fitnessClassifications))))/(43/4) + min((np.mean(listsOfCorrectAccuracy2) - np.mean(listsOfContrastingAccuracy2)),(np.mean(listsOfCorrectAccuracy) - np.mean(listsOfContrastingAccuracy))) - (hamming_distance(fitnessClassificationsOtherSegment, fitnessClassifications)/(43/2))
        return finalFitness,(min((np.sum(fitnessClassificationsOtherSegment)),(np.sum(fitnessClassifications)))),min((np.mean(listsOfCorrectAccuracy2) - np.mean(listsOfContrastingAccuracy2)),(np.mean(listsOfCorrectAccuracy) - np.mean(listsOfContrastingAccuracy))) 


'''Randomly assign a region configuration to begin the optimization
'''
def initial_solution(region_number):
    return random.sample(range(90), region_number)


'''Optimization: this is how regions are changed and selected for convergence. For our analysis, we did 10 regions, and thus swapped 10-20%, being 1-2 regions.
Larger numbers of regions being swapped will increase exploration, but be careful.
'''
def generate_neighbor(solution):
    neighbor = solution.copy()
    num_replacements = random.randint(1, 2)

    for _ in range(num_replacements):
        replace_index = random.randint(0, len(neighbor) - 1)
        new_value = random.randint(0, 89)
        while new_value in neighbor:
            new_value = random.randint(0, 89)
        neighbor[replace_index] = new_value
        
    return neighbor


'''Acceptance function of Simulated Annealing: this evaluates whether the Simulated Annealing should pick an equal or less optimal solution
to explore forwards. Can change this to a different constant to increase/decrease exploration or function altogether for changes in progression of exploration.
'''
def acceptance_probability(old_cost, new_cost, temperature):
    if new_cost > old_cost:
        return 1.0
    else:
        return math.exp((new_cost - old_cost) / (temperature * 0.05))


'''Simulated Annealing algorithm: the pseudocode for this algorithm is available in the Readme. Change temperature and cooling-rate to alter the exploration.
Number of iterations it must reach until termination with no optimal solution is 1000. 
'''
def simulated_annealing(region_number, session1Segment1, session2Segment1, session1Segment2, session2Segment2):
    # print("running SA")
    allSolutions = []
    current_solution = initial_solution(region_number)
    current_cost,_,_ = evaluate_solution(current_solution, session1Segment1, session2Segment1, session1Segment2, session2Segment2)
    temperature = 1.5
    cooling_rate = 0.99 # Adjusted cooling rate
    min_temperature = 0.01
    iteration = 1
    no_improvement_iters = 0
    max_no_improvement_iters = 1000  # Convergence threshold
    bestCost = 0
    bestSolution = []
    bestDiff = 0
    bestClassif = 0
    
    while temperature > min_temperature and no_improvement_iters < max_no_improvement_iters:
        #print("temperature:",temperature)
        new_solution = generate_neighbor(current_solution)
        new_cost, new_classif, diff_cost = evaluate_solution(new_solution, session1Segment1, session2Segment1, session1Segment2, session2Segment2)
        if acceptance_probability(current_cost, new_cost, temperature) > random.random():
            if new_cost > current_cost:
                no_improvement_iters = 0  # Reset counter if improvement
            # print("current solution:",new_solution,"with cost:",new_cost,"classif:",new_classif,"differential:",diff_cost,"temperature:",temperature)
            current_solution = new_solution
            current_cost = new_cost
            if current_cost > bestCost:
                bestCost = current_cost
                bestSolution = current_solution
                bestDiff = diff_cost
                bestClassif = new_classif
            allSolutions.append((current_solution, new_classif, diff_cost))
        else:
            no_improvement_iters += 1  # Increment counter if no improvement

        temperature *= cooling_rate
        iteration += 1


'''This runs a single Simulated Annealing and collects all the regions in sub-networks, performance and classification values extracted from the convergence.
'''
def run_single_SA(iteration, region_number, session1Segment1, session2Segment1, session1Segment2, session2Segment2):
    try:
        finalOutputForStorage = simulated_annealing(region_number, session1Segment1, session2Segment1, session1Segment2, session2Segment2)
        logging.info(f'Iteration {iteration} completed successfully')
    except Exception as e:
        logging.error(f'Error in iteration {iteration}: {e}')
    return finalOutputForStorage


'''This parallelizes the Simulated Annealing across multiple cores on your computer. Each Simulated Annealing will run on one core.
Careful with mp.cpu_count() if you are using cores on your PC for something else; it will use all available cores.
'''
def parallel_SA_runs(num_runs, region_number, session1Segment1, session2Segment1, session1Segment2, session2Segment2):
    with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
        futures = [executor.submit(run_single_SA, i, region_number, session1Segment1, session2Segment1, session1Segment2, session2Segment2) for i in range(num_runs)]
        result_list = []
        for future in as_completed(futures):
            try:
                result_list.append(future.result())
            except Exception as e:
                print(f"Error in worker process: {e}")
    return result_list


'''This allows the Simulated Annealing to be run from the command-line. Shell code is available in another file in this repository.
'''
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("num_iterations", type = int)
    parser.add_argument("region_number", type = int)
    parser.add_argument("output_filename")
    args = parser.parse_args()
    logging.basicConfig(filename='job_log.log'+args.output_filename, level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
    region_number = args.region_number
    num_iterations = args.num_iterations
    listOfOutputs = parallel_SA_runs(num_iterations, region_number, session1Segment1, session2Segment1, session1Segment2, session2Segment2)
    storArr = np.array(listOfOutputs, dtype=object)
    np.save('allRuns'+args.output_filename+'.npy', storArr)

if __name__ == '__main__':
    main()

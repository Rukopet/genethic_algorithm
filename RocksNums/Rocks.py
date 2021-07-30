import random

from typing import List
from copy import deepcopy

from matplotlib import pyplot as plt

FITNESS_LIST_STANDARD = [50, 32, 232, 236]
# FITNESS_LIST_STANDARD = [8, 12, 16, 20]
# kek = [[2,2,2,2],[3,3,3,3],[4,4,4,4],[5,5,5,5]]

POPULATION_SIZE = 200  # количество индивидуумов в популяции

P_CROSSOVER = 0.9  # вероятность скрещивания
P_MUTATION = 0.1  # вероятность мутации индивидуума
MAX_GENERATIONS = 1000  # максимальное количество поколений

IN_DEEP = 5
IN_WIDTH = 4

RANDOM_SEED = 20
random.seed(RANDOM_SEED)


def first_generation() -> List[list]:
    return [[[random.randint(-200, 200) for _ in range(IN_DEEP)]
             for _ in range(IN_WIDTH)]
            for _ in range(POPULATION_SIZE)]


def getFitnessIndividual(individ: list) -> int:
    return sum([abs(sum(ind) - i) * -1 for i, ind in zip(FITNESS_LIST_STANDARD, individ)])


def getFitnessPopulation(population: list) -> List[int]:
    return [getFitnessIndividual(individ) for individ in population]


def clone(individ) -> List[list]:
    return deepcopy(individ)


def setTournament(population: list) -> List[list]:
    ret = []
    p_len = len(population)
    for _ in range(p_len):
        ind1 = ind2 = ind3 = 0
        while ind1 == ind2 or ind2 == ind3 or ind3 == ind1:
            ind1, ind2, ind3 = random.randint(0, p_len - 1), random.randint(0, p_len - 1), random.randint(0, p_len - 1)

        ret.append(deepcopy(max(population[ind1], population[ind2], population[ind3])))
    return ret


def exchangeGenes(parent1: list, parent2: list):
    wrap = random.randint(2, len(parent1) - 1)
    parent1[:wrap], parent2[:wrap] = parent2[:wrap], parent1[:wrap]


def mutation(individ, propability):
    for i in individ:
        for j in i:
            if random.random() < propability:
                j = random.randint(-200, 200)


iteration = 0
max_fittness = -1000
population = first_generation()

maxFitnessValues = []
meanFitnessValues = []

while iteration < MAX_GENERATIONS and max_fittness < 0:
    iteration += 1
    winner_tournament = setTournament(population)
    for parent1, parent2 in zip(winner_tournament[::2], winner_tournament[1::2]):
        if random.random() < P_CROSSOVER:
            exchangeGenes(parent1, parent2)

    for mutant in winner_tournament:
        if random.random() < P_MUTATION:
            mutation(mutant, 1.0 / IN_DEEP * IN_WIDTH)

    tmp_fit = getFitnessPopulation(winner_tournament)
    max_value, index_max_value = max([(v, i) for i, v in enumerate(tmp_fit)])

    mean = max_value / POPULATION_SIZE
    maxFitnessValues.append(max_value)
    meanFitnessValues.append(mean)
    print(f"Поколение {iteration}: Макс приспособ. = {max_value}, Средняя приспособ.= {mean}")
    print("Лучший индивидуум = ", *population[index_max_value], "\n")
    print(f"\t\t\t[ {sum(population[index_max_value][0])} ]\t[ {sum(population[index_max_value][1])} ]\t[ {sum(population[index_max_value][2])} ]\t[ {sum(population[index_max_value][3])} ]")

plt.plot(maxFitnessValues, color='red')
plt.plot(meanFitnessValues, color='green')
plt.xlabel('Поколение')
plt.ylabel('Макс/средняя приспособленность')
plt.title('Зависимость максимальной и средней приспособленности от поколения')
plt.savefig("new")
plt.show()


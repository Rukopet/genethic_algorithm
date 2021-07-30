import unittest

import numpy as np

from RocksNums.Rocks import first_generation, POPULATION_SIZE, IN_WIDTH, IN_DEEP, exchangeGenes


class TestStringMethods(unittest.TestCase):
    def test_first_generation(self):
        self.assertEqual(len(first_generation()), POPULATION_SIZE)

    def test_nested_lists(self):
        # self.assertTrue(np.sum([len(x) == IN_WIDTH for x in first_generation()]) == POPULATION_SIZE)
        self.assertEqual(np.sum([len(x) == IN_WIDTH and len(y) == IN_DEEP for x in first_generation() for y in x]),
                         IN_WIDTH * POPULATION_SIZE)

    def test_cxOnePoint(self):
        parent1 = [[0 for _ in range(IN_DEEP)]for _ in range(IN_WIDTH)]
        parent2 = [[1 for _ in range(IN_DEEP)]for _ in range(IN_WIDTH)]
        exchangeGenes(parent1, parent2)
        print(parent1)
        print(parent2)


if __name__ == '__main__':
    unittest.main()

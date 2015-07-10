'''
unit test for ScoringDict


@author: Markus Rempfler
'''

from numpy_utils import ScoringDict
import unittest

import numpy as np
__updated__ = '10.07.2015'


KEYS = ['precision', 'recall', 'dice', 'prauc', 'rocauc', 'accuracy']


class Test(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_init(self):
        '''simple test cases for `init` and `compute_scores`.
        '''

        # empty init.
        assert np.all(x is None for x in ScoringDict())

        np.random.seed(1337)
        # all-good scores.
        pred = np.random.randn(25) * 0.3 + 0.5
        labels = pred > 0.5
        scores = ScoringDict(labels=labels, predictions=pred)
        for key in KEYS:
            assert scores[key] == 1.

        # second test case with pre-calculated scores.
        # given the seed for randn, this should always yield the same.
        pred = np.array([0, 0.4, 0.6, 1, 0.5])
        labels = np.array([0, 1, 1, 1, 0])
        scores = ScoringDict(labels=labels, predictions=pred)

        self.assertAlmostEqual(scores['accuracy'], 0.8)
        self.assertAlmostEqual(scores['dice'], 0.857, places=2)
        self.assertAlmostEqual(scores['precision'], 0.75)
        self.assertAlmostEqual(scores['recall'], 1.0)

    def test_static(self):
        '''check header layout for printing.
        '''
        np.random.seed(1337)
        self.assertEquals('precision   recall      dice        ' +
                          'prauc       rocauc      accuracy    ' +
                          'threshold   ',
                          ScoringDict.header())
        self.assertEquals('100.00      60.00       75.00       ' +
                          '81.27       72.00       80.00       ' +
                          '20.67       ',
                          str(ScoringDict(predictions=np.random.randn(10),
                                          labels=np.random.randn(10) > 0)))

        # now the "hard" prediction case.
        labels = np.array([0, 1, 1, 1, 0])
        pred = np.array([0, 1, 1, 1, 1])
        scores = ScoringDict(labels=labels, predictions=pred, soft=False)
        self.assertAlmostEqual(scores['accuracy'], 0.8)
        self.assertAlmostEqual(scores['dice'], 0.857, places=2)
        self.assertAlmostEqual(scores['precision'], 0.75)
        self.assertAlmostEqual(scores['recall'], 1.0)
        self.assertIsNone(scores['rocauc'])
        self.assertIsNone(scores['prauc'])
        self.assertEquals('75.00       100.00      85.71       ' +
                          '-           -           80.00       ' +
                          '-           ',
                          str(scores))


if __name__ == "__main__":
    unittest.main()

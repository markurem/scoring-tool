'''scoring dict class for evaluation of binary predictors.

@author: Markus Rempfler
'''

__updated__ = '10.07.2015'

from numpy import alltrue, argmax, mean
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score

# TODO:
# * write an optional formatter that outputs latex code for table rows.
#


class ScoringDict(dict):

    '''Extension to dict for scoring predictions in a systematic way.
    '''

    # scores.
    _keys = ['precision',
             'recall',
             'dice',        # alias F1 score.
             'prauc',       # area under the precision recall curve.
             'rocauc',      # area under the receiver operator characteristic.
             'accuracy',
             'threshold',   # F1 optimal threshold.
             ]
    _fmt = '{:12}' * len(_keys)

    # print header.
    @staticmethod
    def header():
        '''prints all keys in the same order as print(self)
        '''
        return ScoringDict._fmt.format(*[key for key in ScoringDict._keys])

    def __init__(self,
                 labels=None,
                 predictions=None,
                 soft=True):
        '''constructor.

        makes sure that all score-keys are set and, if labels and predictions
        are given, directly computes its scores.

        Parameters
        ----------
        labels : array_like
            ground truth labels.
        predictions : array_like
            predicted values. Need to have the same shape as `labels`.
        soft : bool, optional (default: True)
            expect a soft prediction (ie. probabilities).

        '''

        super(ScoringDict, self).__init__()
        for key in self._keys:
            self[key] = None

        if labels is None or predictions is None:
            return

        assert alltrue(labels.shape == predictions.shape)
        self.compute_scores(labels, predictions, soft=soft)

    def __str__(self):
        '''prints all scores in percentages.
        '''
        return self._fmt.format(*['%2.2f' % (100 * self[key])
                                  if self[key] is not None else '-'
                                  for key in self._keys])

    def compute_scores(self, labels, predictions, soft=True):
        '''compute all scores as efficiently as possible.

        Parameters
        ----------
        labels : array_like, shape=[n_samples,]
            ground truth labels.
        predictions : array_like, shape=[n_samples,]
            predictions to be scored.
        soft : bool, optional (default=True)
            if `soft=True`, a probabilistic prediction is expected and the
            dice optimal threshold is determined.

        Notes
        -----
        This method was made for binary classes and should be complemented
        with a label binarizer if multi class problems are adressed.
        '''
        if soft:
            precision, recall, thresholds = precision_recall_curve(
                labels,
                predictions)

            # compute area-under-the curve for precision-recall and ROC
            self['prauc'] = auc(recall, precision)
            self['rocauc'] = roc_auc_score(labels, predictions)

            # now turn soft predictions into hard ones by taking the Dice
            # optimal threshold.
            ind, _ = compute_dice_opt_threshold(
                precision,
                recall)
            self['threshold'] = thresholds[ind]
            predictions = predictions >= self['threshold']

        # TODO: handle the multiclass case.
        self['precision'], self['recall'], self['dice'], self['support'] = \
            precision_recall_fscore_support(
            labels,
            predictions,
            average='binary')
        self['accuracy'] = mean(predictions == labels)


def compute_dice_opt_threshold(precision, recall):
    '''Computes the F1 optimal operating point.

    This is a utility function for precision_recall_curve.
    '''

    # ref: scikit-learn.org // sklearn.metrics.f1_score
    dice = 2 * (precision * recall) / (precision + recall)
    ind = argmax(dice)
    return ind, dice[ind]

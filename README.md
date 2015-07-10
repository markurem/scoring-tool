# scoring-tool

Simple scoring class for python.

## Requirements

numpy
sklearn

## Example

```
>>> labels = ... # some labels
>>> prediction = clf.predict(data)  # some classifier
>>> scores = ScoringDict(labels=labels.flatten(), predictions=prediction.flatten())

...

>>> print ScoringDict.header()
>>> print scores
```

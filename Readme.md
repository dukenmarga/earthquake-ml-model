
### Download dataset
Source: https://github.com/smousavi05/STEAD

Make sure we have
- merge.csv
- merge.hd5

### Dataset Preparation
Preparation dataset:
```
python3 convert.py
python3 convert-for-testing.py
python3 sdof-analysis.py
python3 sdof-analysis-for-testing.py
```

### Training
```
python3 training-random-forest.py
```
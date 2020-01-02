# EE369-Kaggle
##### Code Structure:

```
|── README.md
|── data
│   ├── test 											// test data set
│   │   ├── candidate100.npz
│   |   ├── candidateXX.npz
│   │   └── test.csv
│   └── train_and_valid						// train and valid data set
│       ├── candidate1.npz
│       ├── candidateXX.npz
│       ├── train.csv
│       └── val.csv
├── mylib
│   ├── __init__.py
│   ├── dataloader
│   │   ├── ENVIRON								// Set training data and testing data path
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   └── path_manager.py
│   ├── models
│   │   ├── __init__.py
│   │   ├── densenet.py
│   │   ├── densesharp.py
│   │   ├── losses.py
│   │   ├── metrics.py
│   │   └── misc.py
│   └── utils
│       ├── __init__.py
│       ├── misc.py
│       ├── multicore.py
│       └── plot3d.py
├── result
│   └── result.npy
├── submission.csv							// Submission.csv to upload
├── test.py											// python test.py
├── train.py										
└── weight.h5									  // Model weight
```



##### Data Path Configuration

1. Set train, test dataset path

Change the setting in `./mylib/dataloader/ENVIRON` 

```
{
  "TRAIN_DATASET": "data/train_and_valid",	// The training data path
  "TEST_DATASET": "data/test",							// The testing data path
  																					// (contains test.csv and test data)
  "VAL_DATASET": "data/train_and_valid"
}
```

2. Model weight path : `./weight.h5` 

3. Result path: `./submission.csv` 



##### Run Code:

1. Run test code

```python
python test.py
```






# NodeSentry: HPC Node Anomaly Detection Model Based on Transformer and TSFEL

## Project Introduction

The framework encompasses two principal phases: the offline model training phase and the online anomaly detection phase. During the offline model training phase, we first preprocess the raw MTS. We then perform coarse-grained clustering on the processed MTS, which enables the model to train on the clusters rather than on individual MTS segments. Next, we assign expert networks to different segments within the cluster for model sharing. In the online anomaly detection phase, we extract features from preprocessed MTS to match patterns in the cluster library. Subsequently, the appropriate model is dynamically assigned to identify anomalies in the target nodes.

## Directory Structure


```
NodeSentry-model 
├── node_data                    # Directory for storing time series data
├── clustering         # Directory for coarse-grained clustering methods and results
│   ├── classify_result            # Directory for storing clustering results
│   ├── feature_center             # Directory for storing cluster centers and feature weights
│   ├── normalized_result          # Directory for storing standardized feature indices 
│   ├── clustering.py              # Coarse-grained clustering code
│   └── utils.py                   # Utility functions for clustering
├── model_sharing      # Directory for fine-grained model sharing methods and results
│   ├── model                      # Directory for storing trained models
│   ├── result                     # Directory for storing anomaly detection results
│   ├── detection.py               # Anomaly detection code
│   ├── training.py                # Fine-grained model sharing training code
│   ├── TransformerAndMoe.py       # Transformer model training and detection related function
│   └── utils.py                   # Utility functions for data processing and model sharing
├── cluster.sh                     # Coarse-grained clustering command
├── config.yml                     # Configuration file for setting parameters and file paths
├── detect.sh                      # Anomaly detection command
├── requirements.txt               # Environment packages
└── train.sh                       # Fine-grained model sharing training command
```

## Data Description

### 1. Job scheduling list

A job scheduling list is required during anomaly detection to obtain the job status of a specific node. The format of the job list is as follows:

```
NodeList|Start|End|State
node1|2024-01-01T00:00:00|2024-01-02T00:00:00|CANCELLED+
node2, node3|2024-01-02T00:00:00|2024-01-03T00:00:00|COMPLETED
```
### 2. Data Format

The data in the `node_data` folder is used in this project for coarse-grained clustering, fine-grained model sharing training, and anomaly detection. This data is stored as a T-row M+1 column CSV matrix, where T is the time length and M represents the number of metrics. There is an additional column for the timestamp, as shown below:

```
timestamp, feature1, feature2, feature3
2024-01-01 00:00:00, 0.5, 0.7, 0.2
2024-01-01 00:01:00, 0.6, 0.8, 0.3
...
```

## Installation Steps

1. Download the project
2. Install dependencies:

```
pip install -r requirements.txt
```

Note: Python version is 3.8.10

3. **Configuration before use**:

- **config.yml**: Configuration file containing parameters and paths for data processing and model training.
- **metric.json**: Metrics file containing the metrics involved in training.
- **node_data**: Data folder storing MTS segments for coarse-grained clustering, fine-grained model sharing training, and anomaly detection.
- **jobinfo**: Job scheduling list, including job state information.

## How to Use

1. **Data Preparation:** Prepare MTS and configuration files. The data folder should be structured as follows:

```
node_data                   # Directory for storing MTS
├── offline                 # Training set 
│   ├── job                  # job execution
│   │   └── node1            
│   │   │   └── metric_job1.csv 
│   └── nojob                # idle waiting
│   │   └── node1.csv 
└── online                  # Test set     
	└── node1                # Node 1 test data     
	└── metric.csv
```  

2. **Coarse-grained Clustering:** For example, execute `python clustering.py --job_num 20 --nojob_num 10 --node_num 10` to perform coarse-grained clustering. This will output the clustering results, cluster centers and feature indices for each cluster, feature weights, etc. We have encapsulated this command into the `cluster.sh` and can directly execute `sh cluster.sh`.
3. **Fine-grained Model Sharing Training:** Execute `python training.py --max_epochs 30 --batch_size 50 --window_size 20 --nhead 3 --num_layers 3 --num_experts 3 --k 1` to perform fine-grained model sharing training and save the trained model. We have encapsulated this command into the `train.sh` and can directly execute `sh train.sh`.
4. **Anomaly Detection:** Execute `python detection.py --interval 10 --match_len 200` to perform anomaly detection. This will output the model reconstructed data and anomaly score for each node. We have encapsulated this command into the `detect.sh` and can directly execute `sh detect.sh`.
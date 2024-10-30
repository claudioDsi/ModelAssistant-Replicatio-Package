# ModelAssistant-Replication-Package

## Overview
This repository contains the replication package for the article entitled "Using graph-based structures in intelligent modeling assistants: an
experience report"

## Features
- The repo is structured as follows. 
- The folder *MORGAN* contains the source code of the tool used in the paper
- For the BORA tool please refer to the corresponding repository available [here](https://github.com/iliriani/BORA_Ecore)
- The dataset used in the evaluation are stored in the *Dataset.zip* folder  



## Installation
To run MORGAN please follows the following steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/claudioDsi/ModelAssistant-Replication-Package.git
   ```
2. Navigate to the project directory:
   ```bash
   cd ModelAssistant-Replication-Package
   ```
3. Install the dependencies from the requirement.txt file:
   ```bash
   pip install -r /path/to/requirements.txt
   ```

**Please note** that Python 3.7 is required for the Grakel library. 

## Usage
To run MORGAN, you need to run the following steps:

```bash
python main.py data_path n_classes n_items size rec_type
```
where:  

- data_path: (string) Path to the dataset folder containing the train and test files.
- n_classes: (integer) Number of classes for recommendation.
- n_items: (integer) Number of items to process for each recommendation.
- size: (integer) Size of the test according to different configurations
- rec_type: (string) Type of recommendation (class or attrs)


To compute the similarity metrics, you can use the **compute_similarity** function in the **main.py** by specifying the source data, i.e., one of the three dataset contained in teh zip file, and the output CSV name. 





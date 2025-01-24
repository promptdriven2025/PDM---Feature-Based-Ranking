# Feature Based Ranking

## Overview
This repository contains scripts and utilities for selecting, ranking, and optimizing SEO texts using machine learning models. The system processes input text datasets, extracts features, and ranks content based on predefined models.

## File Breakdown
### **Primary Scripts**
#### 1. `choose_seo_texts_new.py`
- **Purpose:** Reads working set files, extracts features, normalizes them, and generates output ranking files.
- **Inputs:**
  - `working_set_file_path` (CSV): Contains initial document rankings.
  - `features_dir` (directory): Directory with feature files.
  - `feature_list` (list): Features used for ranking.
- **Outputs:**
  - Normalized feature scores.
  - Ranking prediction output files.

#### 2. `choose_seo_texts_new_java.py`
- **Purpose:** Similar to `choose_seo_texts_new.py`, but utilizes a Java-based ranking model.
- **Inputs:**
  - `g_data.csv` or `t_data.csv` for processing.
  - Java RankLib model.
- **Outputs:**
  - Ranked document lists.

#### 3. `create_bot_features.py`
- **Purpose:** Generates feature files for ranking, including semantic similarity and query-term frequency calculations.
- **Inputs:**
  - Ranked document lists.
  - Document text collections.
  - Word embeddings for similarity calculations.
- **Outputs:**
  - Feature matrices for ranking models.

#### 4. `create_bot_qrels.py`
- **Purpose:** Generates query relevance (QREL) files for ranking model evaluation.
- **Inputs:**
  - Ranked document lists.
- **Outputs:**
  - QREL files in standard evaluation format.

#### 5. `ranking_logic_example.py`
- **Purpose:** Implements the ranking pipeline, including index merging and executing ranking models.
- **Inputs:**
  - Indri index paths.
  - Working set files.
- **Outputs:**
  - Ranked document lists.

### **Utility Modules**
#### 6. `utils.py`
- **Contains:**
  - File processing utilities.
  - Index creation and merging functions.
  - Query and document parsing functions.

#### 7. `gen_utils.py`
- **Contains:**
  - General utilities for multiprocessing and bash command execution.

#### 8. `vector_functionality.py`
- **Contains:**
  - Functions for text similarity calculations.
  - Cosine similarity, centroid calculations, and document vector processing.

## Installation & Usage
### **Installation**
Ensure the following dependencies are installed:
```bash
pip install pandas numpy tqdm lxml krovetzstemmer
```
### **Usage**
#### **Generating Features and Ranking Documents**
```bash
python choose_seo_texts_new.py
```
Or using Java-based ranking:
```bash
python choose_seo_texts_new_java.py
```
#### **Generating Feature Files**
```bash
python create_bot_features.py
```
#### **Generating QRELs for Evaluation**
```bash
python create_bot_qrels.py
```
#### **Running the Ranking Model**
```bash
python ranking_logic_example.py
```

## Dependencies
- Python 3.x
- Java 1.8+ (for Java-based ranking models)
- External dataset files (`g_data.csv`, `t_data.csv`)

## Notes
- Ensure all required files (listed in the scripts) are available before running.
- Modify paths in scripts as necessary to fit your directory structure.
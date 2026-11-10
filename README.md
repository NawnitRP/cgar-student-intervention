# CGAR: Confidence-Gated Actionable Recommendation

Code for our paper "Confidence-Gated Actionable Intervention Recommendation for At-Risk
Students" (Nawnit Kumar and Kapil Madan). Running it regenerates the tables and figures
reported in the paper.

## Files
- code/cgar.py - the hybrid MLP-XGBoost predictor, the conformal gate and the recommender
- code/datasets.py - loaders for the three datasets
- code/run_experiments.py - runs all experiments and saves tables and figures
- code/tests/test_cgar.py - tests for the conformal and recommendation parts
- run_all.bat, run_uci.bat - run everything / only the UCI data (Windows)

## Setup
Python 3.10 or newer:

    pip install numpy pandas scipy scikit-learn matplotlib xgboost

## Data
The data is not included. Download it from the original sources:
- UCI Student Performance: https://archive.ics.uci.edu/dataset/320/student+performance (put in data/uci)
- xAPI-Edu-Data: https://www.kaggle.com/datasets/aljarah/xAPI-Edu-Data (put in data/xapi)
- OULAD: https://analyse.kmi.open.ac.uk/open-dataset (put in data/oulad)

## Run
On Windows, double-click run_all.bat (a full run takes a few hours). Results are written
to paper/results.

## License
MIT. The datasets have their own licenses, so please cite the original authors.

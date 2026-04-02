We conduct experiments comparing BM25 Atire to MDRL and MDRM on mLLeQA dataset. 

The dataset code and results related to MDRL are available at the following link : 
https://zenodo.org/records/17436378

The same datasets are used to run experiments on MDRM and BM25 as well.

To run MDRM experiments for dutch. 
1) Navigate to mdrm/dutch
2) Download dutch/data from https://zenodo.org/records/17436378 in the current directory
3) Run:  
   $ bash scripts/run_biencoder_training.sh 
4) The environment must be setup according to results/mdrm/dutch/version_details

Repeat for all languages. 

To run BM25 experiments for dutch : 
1) Navigate to bm25/dutch
2) Download dutch/data from https://zenodo.org/records/17436378 in the current directory
3) Run:  
   $ python3 lexical_experiments/gridsearch_bm25.py  
4) The environment must be setup according to results/bm25/dutch/version_details

The repository with code, datasets and results for cross-lingual and backtranslation experiments is provided at 
https://github.com/Aln2004/english-biencoder-eval

Set all paths appropriately should any errors arise

Results for MDRM and BM25 experiments can be found in the results/bm25 and results/mdrl folders. 

To replicate the experiments the outputs as obtained from : 
    pip list
    pip3 list
    pip freeze
    pip3 freeze
    conda env export --from-history 
    conda list
    conda list --explicit

are present in the version details folder in each results folder of each language. 


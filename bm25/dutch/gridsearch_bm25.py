import datetime
print("Doing Dutch BM25. Time now is ", datetime.datetime.now())

import os
print("Current working directory is ", os.getcwd())
import sys
import pathlib
import nltk

try:
    from src.data.lleqa import LLeQADatasetIRLoader
except ModuleNotFoundError:
    sys.path.append(str(pathlib.Path().resolve()))
    from src.data.lleqa import LLeQADatasetIRLoader

from src.utils.SentenceTransformer import InformationRetrievalEvaluatorCustom
from sentence_transformers import util

import importlib.metadata
import rank_bm25
import pickle

from rank_bm25 import BM25Okapi
print("The version of rank_bm25 used is : ", importlib.metadata.version("rank_bm25"))

import inspect
print("The source code of a class of BM25Okapi is below : ")
print(inspect.getsource(BM25Okapi))

def dev_bm25(dev_evaluator, k1, b, epsilon, exp_type):

    print("Dumping the InformationRetrievalEvaluator constructed on the dev set to: lexical_experiments/bm25_pickle_output/" + exp_type + "/devevaluatorout.pickle")
    with open("lexical_experiments/bm25_pickle_output/" + exp_type + "/devevaluatorout.pickle", "wb") as dev_eval_file:
        pickle.dump(dev_evaluator, dev_eval_file)

    print("Tokenizing Corpus and initialising BM25Okapi Ranker")

    tokenized_corpus = [doc.split() for doc in dev_evaluator.corpus]
    print("Values input to BM25Okapi for dev set evaluation are as follows : ")
    print("k1 = ", k1)
    print("b = ", b)
    print("epsilon = ", epsilon)
    bm25 = BM25Okapi(tokenized_corpus, k1=k1, b=b, epsilon=epsilon)
    queries_result_list = []

    print("Executing BM25Okapi on all queries")
    for query_text in dev_evaluator.queries:
        scores = bm25.get_scores(query_text.split())
        query_results = [
            {
            "corpus_id": dev_evaluator.corpus_ids[i],
                        "score": float(scores[i])
        }
        for i in range(len(scores))
        ]
        queries_result_list.append(query_results)

    print("Dumping the inferences output by BM25 Okapi to: lexical_experiments/bm25_pickle_output/" + exp_type + "/bm25okapiinferencesout.pickle")
    with open("lexical_experiments/bm25_pickle_output/"+ exp_type +"/bm25okapiinferencesout.pickle", "wb") as inferences_eval_file:
        pickle.dump(queries_result_list, inferences_eval_file)

    print("Running compute_metrics of an instance of the InformationRetrievalEvaluator built on the dev set to calculate retrieval scores ")
    results = dev_evaluator.compute_metrics(queries_result_list)

    return results

def test_bm25(test_evaluator, k1, b, epsilon, exp_type):
    # Check Name parameter
    print("Dumping the InformationRetrievalEvaluator constructed on the test set to: lexical_experiments/bm25_pickle_output/" + exp_type + "/testevaluatorout.pickle")
    with open("lexical_experiments/bm25_pickle_output/" + exp_type + "/testevaluatorout.pickle", "wb") as test_eval_file:
        pickle.dump(test_evaluator, test_eval_file)

    print("Tokenizing Corpus and initialising BM25Okapi Ranker")

    tokenized_corpus = [doc.split() for doc in test_evaluator.corpus]
    print("Values input to BM25Okapi for test set evaluation are as follows : ")
    print("k1 = ", k1)
    print("b = ", b)
    print("epsilon = ", epsilon)
    bm25 = BM25Okapi(tokenized_corpus, k1=k1, b=b, epsilon=epsilon)
    queries_result_list = []

    print("Executing BM25Okapi on all queries")
    for query_text in test_evaluator.queries:
        scores = bm25.get_scores(query_text.split())
        query_results = [
            {
            "corpus_id": test_evaluator.corpus_ids[i],
                        "score": float(scores[i])
        }
        for i in range(len(scores))
        ]
        queries_result_list.append(query_results)

    print("Dumping the inferences output by BM25 Okapi to: lexical_experiments/bm25_pickle_output/" + exp_type + "/bm25okapiinferencesout.pickle")
    with open("lexical_experiments/bm25_pickle_output/"+ exp_type +"/bm25okapiinferencesout.pickle", "wb") as inferences_eval_file:
        pickle.dump(queries_result_list, inferences_eval_file)

    print("Running compute_metrics of an instance of the InformationRetrievalEvaluator built on the test set to calculate retrieval scores ")
    results = test_evaluator.compute_metrics(queries_result_list)

    return results

def main():

    data = LLeQADatasetIRLoader(
        stage='fit',
        corpus_path_or_url="data/lleqa/dutch_articles.json",
        train_path_or_url="data/lleqa/dutch_questions_train.json",
        dev_path_or_url="data/lleqa/dutch_questions_val.json",
        test_path_or_url="data/lleqa/dutch_questions_test.json",
        negatives_path_or_url="data/lleqa/negatives/negatives_bm25.json",
    ).run()

    print("Dumping the data variable to: lexical_experiments/bm25_pickle_output/datasetout.pickle")
    with open("lexical_experiments/bm25_pickle_output/datasetout.pickle", "wb") as dataset_file:
        pickle.dump(data, dataset_file)

    print("Staring Dataset checks")
    print("Queries in the test set are : ")
    print(data["test_queries"])
    print()
    print("Labels in the test set are : ")
    print(data["test_labels"])
    print()
    print("Queries in the dev set are : ")
    print(data['dev_queries'])
    print()
    print("Labels in the dev set are : ")
    print(data['dev_labels'])
    print()
    print("A single element of the corpus is : ")
    print(data["corpus"][1])
    print()
    print("A single element of the train set is : ")
    print(data["train"][2])
    print()
    print("Ended Dataset Checks")

    test_evaluator = InformationRetrievalEvaluatorCustom(
        name=f'dataset_test', queries=data['test_queries'], relevant_docs=data['test_labels'], corpus=data['corpus'],
        precision_recall_at_k=[1, 5, 10, 20, 50, 100, 200, 500], map_at_k=[10, 100], mrr_at_k=[10, 100], ndcg_at_k=[10, 100],
        score_functions={"cos_sim":getattr(util, "cos_sim")},
        log_callback=None, show_progress_bar=None,
        corpus_chunk_size=None, batch_size=None,
    )

    dev_evaluator = InformationRetrievalEvaluatorCustom(
        name=f'dataset_dev', queries=data['dev_queries'], relevant_docs=data['dev_labels'], corpus=data['corpus'],
        precision_recall_at_k=[1, 5, 10, 20, 50, 100, 200, 500], map_at_k=[10, 100], mrr_at_k=[10, 100], ndcg_at_k=[10, 100], accuracy_at_k=[1],
        score_functions={"cos_sim": getattr(util, "cos_sim")},
        log_callback=None, show_progress_bar=None,
        corpus_chunk_size=None, batch_size=None,
    )

    potential_k1 = [0.5, 0.9, 1.2, 1.5, 1.8, 2.0]
    potential_b = [0.0, 0.25, 0.5, 0.75, 1.0]
    potential_epsilon = [0.0, 0.25]

    result_dict = {}
    for k1 in potential_k1:
        for b in potential_b:
            for epsilon in potential_epsilon:
                exp_type = f"dev_data/k1={str(k1).replace('.','p')}_b={str(b).replace('.','p')}_e={str(epsilon).replace('.','p')}"
                if not os.path.exists("lexical_experiments/bm25_pickle_output/" + exp_type):
                    os.makedirs("lexical_experiments/bm25_pickle_output/" + exp_type)
                print("Beginning execution of BM25Okapi on dev set with k1 = ", k1, " b = ", b, " epsilon = ", epsilon)
                print("All relevant files corresponding to running BM25Okapi on the dev set with k1 = ", k1, " b = ", b, " epsilon = ", epsilon, " will be stored in ", "lexical_experiments/bm25_pickle_output/" + exp_type)

                results = dev_bm25(dev_evaluator, k1, b, epsilon, exp_type)
                print("Results for dev set on k1 = ", k1, " b = ", b, " epsilon = ", epsilon, " without pretty printing are just below this statement: ")
                print(results)
                print()
                print("Pretty printed results for dev set on k1 = ", k1, " b = ", b," epsilon = ", epsilon, " are just below this statement ")
                for metric, values in results.items():
                  print(f"\n{metric}")
                  print("-" * len(metric))
                  for k in sorted(values):
                    print(f"k={k} {values[k]:.4f}")
                print("----------------------------------------------------------------------------------------------------------")
                result_dict[(k1, b, epsilon)] = results['recall@k'][500]

    print("The results obtained on various values of k1, b and epsilon on the dev set are : ")
    print(result_dict)
    best_k1_b_epsilon = max(result_dict, key=result_dict.get)
    print("The best k1, b, epsilon combination on Recall@500 on the dev set is : ", best_k1_b_epsilon)

    test_k1 = best_k1_b_epsilon[0]
    test_b = best_k1_b_epsilon[1]
    test_epsilon = best_k1_b_epsilon[2]

    print("----------------------------------------------------------------------------------------------------------")

    test_exp_type = f"test_data/k1={str(test_k1).replace('.','p')}_b={str(test_b).replace('.','p')}_e={str(test_epsilon).replace('.','p')}"
    if not os.path.exists("lexical_experiments/bm25_pickle_output/" + test_exp_type):
        os.makedirs("lexical_experiments/bm25_pickle_output/" + test_exp_type)

    print("Beginning execution of BM25Okapi on test set with k1 = ", test_k1, " b = ", test_b, " epsilon = ", test_epsilon)
    print("All relevant files corresponding to running BM25Okapi on the test set with k1 = ", test_k1, " b = ", test_b, " epsilon = ", test_epsilon, " will be stored in ", "lexical_experiments/bm25_pickle_output/" + test_exp_type)
    test_results = test_bm25(test_evaluator, test_k1, test_b, test_epsilon, test_exp_type)
    print("The results of running BM25Okapi with the best parameters on the test set without pretty printing are below : ")
    print(test_results)
    print("The results of running BM25Okapi with the best parameters on the test set with pretty printing are below : ")
    for metric, values in test_results.items():
        print(f"\n{metric}")
        print("-" * len(metric))
        for k in sorted(values):
            print(f"k={k} {values[k]:.4f}")

    print("Starting Common Parameters on test set BM25Okapi")
    common_k1 = 1.5
    common_b = 0.75
    common_epsilon = 0.25

    test_exp_type = f"test_data/k1={str(common_k1).replace('.','p')}_b={str(common_b).replace('.','p')}_e={str(common_epsilon).replace('.','p')}"
    if not os.path.exists("lexical_experiments/bm25_pickle_output/" + test_exp_type):
        os.makedirs("lexical_experiments/bm25_pickle_output/" + test_exp_type)

    print("Beginning execution of BM25Okapi on test set with k1 = ", common_k1, " b = ", common_b, " epsilon = ", common_epsilon)
    print("All relevant files corresponding to running BM25Okapi on the test set with k1 = ", common_k1, " b = ", common_b, " epsilon = ", common_epsilon, " will be stored in ", "lexical_experiments/bm25_pickle_output/" + test_exp_type)
    test_results = test_bm25(test_evaluator, common_k1, common_b, common_epsilon, test_exp_type)
    print("The results of running BM25Okapi with the common parameters on the test set without pretty printing are below : ")
    print(test_results)
    print("The results of running BM25Okapi with the common parameters on the test set with pretty printing are below : ")
    for metric, values in test_results.items():
        print(f"\n{metric}")
        print("-" * len(metric))
        for k in sorted(values):
            print(f"k={k} {values[k]:.4f}")

if __name__ == '__main__':
	main()

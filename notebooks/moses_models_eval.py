__author__ = 'Abdulrahman Semrie<hsamireh@gmail.com>'

from mozi_cross_val.main.model_evaluator import ModelEvaluator
from mozi_cross_val.models.objmodel import MosesModel
import argparse
import pandas as pd
import numpy as np
import re
import os

output_regex_no_complexity = re.compile(r"(-?\d+) (.+)")
output_regex_w_complexity = re.compile(r"(-?\d+) (.+) \[(.+)\]")

def parse_args():
    parser = argparse.ArgumentParser(description="A script to get scores of MOSES Models")
    parser.add_argument("--input", type=str, default='',
                        help="Path to input file to moses")
    parser.add_argument("--combo", type=str, default='',
                        help="Path to file that contains combo programs")
    parser.add_argument("--target", type=str, default='posOutcome',
                        help="Name of the target column in the input file")
    parser.add_argument("--output", type=str,
                        help="Path to save the output file to")
    return parser.parse_args()


def format_combo(combo_file, complexity=False):
    """
    Build model objects from the moses output
    :param combo_file: The path to the raw combo output
    :return:
    """

    models = []
    with open(combo_file, "r") as fp:
        for line in fp:
            if complexity:
                match = output_regex_w_complexity.match(line.strip())
            else:
                match = output_regex_no_complexity.match(line.strip())
            if match is not None:
                model = match.group(2).strip()
                if model == "true" or model == "false":
                    continue
                if complexity:
                    complexity = match.group(3).split(",")[2].split("=")[1]
                    models.append(MosesModel(model, complexity))
                else:
                    models.append(MosesModel(model, None))

    return models

def evaluate_models():
    print("Starting..")
    args = parse_args()
    target_col = args.target
    input_file = args.input
    output_file = args.output
    combo_file = args.combo

    if not os.path.exists(input_file):
        print("Input file %s not found. Please provide the correct path" % input_file)
        return
    if not os.path.exists(combo_file):
        print("Combo file %s not found. Please provide the correct path" % combo_file)
        return
    print("Parsing Models...")
    models = format_combo(combo_file)
    model_eval = ModelEvaluator(target_col)
    print("Evaluating Models...")
    matrix = model_eval.run_eval(models, input_file)
    scores = model_eval.score_models(matrix, input_file)

    for model, score in zip(models, scores):
        model.test_score = score

    cols = ["model", "recall_test", "precision_test"]
    res_matrix = np.empty([len(models), 3])
    for i, model in enumerate(models):
        for j, k in cols:
            res_matrix[i][j] = model[k]
    output_csv = pd.DataFrame(data=res_matrix, columns=["model", "recall", "precision"])
    output_csv.to_csv(output_file, index=False)

    print("Done!")

if __name__ == "__main__":
    evaluate_models()

__author__ = 'Abdulrahman Semrie<hsamireh@gmail.com>'

from mozi_cross_val.main.model_evaluator import ModelEvaluator
from mozi_cross_val.models.objmodel import MosesModel
import argparse
import pandas as pd
import numpy as np
import re
import os
import tempfile
import subprocess

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

def run_eval(self, models, input_file):
        """
        Evaluate a list of model objects against an input file
        :param: models: list of model objects
        :param input_file: the location of the input file
        :return: matrix:
        nxm matrix where n is the number of models and m is the number of samples. the matrix contains the predicted
        output of each model on the sample
        """

        input_df = pd.read_csv(input_file)
        num_samples, num_models = input_df.shape[0], len(models)
        matrix = np.empty((num_samples, num_models), dtype=int)

        temp_eval_file = tempfile.NamedTemporaryFile().name
        eval_log = tempfile.NamedTemporaryFile().name

        for i, moses_model in enumerate(models):
            cmd = ['eval-table', "-i", input_file, "-c", moses_model, "-o", temp_eval_file, "-u",
                   self.target_feature, "-f", eval_log]
            process = subprocess.Popen(args=cmd, stdout=subprocess.PIPE)

            stdout, stderr = process.communicate()

            if process.returncode == 0:
                matrix[,:i] = np.genfromtxt(temp_eval_file, skip_header=1, dtype=int)
            else:
                self.logger.error("The following error raised by eval-table %s" % stderr.decode("utf-8"))
                raise ChildProcessError(stderr.decode("utf-8"))
        return matrix


def score_models(self, matrix, input_file):
        """
        Takes a matrix containing the predicted value of each model and a file to containing the actual target values
        It calculates the accuracy, recall, precision, f1 score and the p_value from mcnemar test of each model
        :param matrix: The input matrix
        containing the predicted values for each model. This the matrix returned by functions like run-eval
        :param input_file: this the test file containing the actual target values
        :return: matrix: returns an nx4 matrix where n is the number of model.
        """
        score_matrix = np.empty([0, 5])

        df = pd.read_csv(input_file)
        target_value = df[self.target_feature].values
        null_value = np.zeros((len(target_value),))

        for i in range(matrix.shape[1]):
            y_pred = matrix[:,i]
            recall, precision, accuracy, f_score = ModelEvaluator._get_scores(target_value, y_pred)
            p_value = ModelEvaluator.mcnemar_test(target_value, y_pred, null_value)
            score_matrix = score_matrix.append(score_matrix, np.array([[recall, precision, accuracy, f_score, p_value]]))

        return score_matrix

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

    res_matrix = np.empty([len(models), 1])
    for model in models:
        res_matrix = np.append(res_matrix, np.array([[model]]))

    df_data = np.concatenate(res_matrix, scores, axis=1)
    output_csv = pd.DataFrame(df_data, columns=["model", "recall", "precision", "accuracy", "f_score", "p_value"])
    output_csv.to_csv(output_file, index=False)

    print("Done!")

if __name__ == "__main__":
    evaluate_models()

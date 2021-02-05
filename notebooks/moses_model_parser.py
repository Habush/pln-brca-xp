__author__ = 'Abdulrahman Semrie<hsamireh@gmail.com>'

from lark import Transformer
from lark import Lark
import argparse
import pandas as pd
import numpy as np
import os

grammar = '''
    combo: func | feature | negate
    func:  and_or lpar param+ rpar
    param: feature | func | negate
    lpar: "("
    rpar: ")"
    and_or: "and" | "or"
    negate: "!" feature
    feature: "$" name
    name: /[^$!()]+/

    %import common.ESCAPED_STRING
    %import common.WS
    %ignore WS
'''

combo_parser = Lark(grammar, start='combo', parser='lalr')


class ComboTreeTransform(Transformer):
    """
    This class builds a feature count dictionary by going through a combo tree
    """

    def __init__(self):
        self.fcount = {}
        self.up = True
        self.curr = None
        self.score = None

    def name(self, s):
        feature = s[0].value.strip()
        self.curr = feature
        if not feature in self.fcount:
            self.fcount[feature] = 0

        self.fcount[feature] = self.fcount[feature] + self.score

    def parse_feats(self, tree, score):
        self.score = score
        self.transform(tree)


def parse_args():
    parser = argparse.ArgumentParser(description="A script to get scores of MOSES Models")
    parser.add_argument("-i" ,"--input", type=str, default='',
                        help="Path to input csv file")
    parser.add_argument("-n", type=int, default=50, help="Number of genes to select (default=50)")
    parser.add_argument("-o", "--output", type=str,
                        help="Path to save the output file to")
    return parser.parse_args()

def get_features():
    args = parse_args()

    if not os.path.exists(args.input):
        print("Input file %s not found. Please provide the correct path" % args.input)
        return

    input_df = pd.read_csv(args.input)
    num = args.n
    tree_transformer = ComboTreeTransform()

    for i, row in input_df.iterrows():
        model = row["model"]
        score = row["f_score"]
        tree = combo_parser.parse(model)
        tree_transformer.parse_feats(tree, score)

    feats_dict =  tree_transformer.fcount
    ranked_genes = sorted(feats_dict, key=feats_dict.get, reverse=True)
    if num > len(feats_dict):
        print("Cannot select {0} genes from a list with {1} elements. Returning all genes".format(num, len(ranked_genes)))
    else:
        ranked_genes = ranked_genes[:num]
    with open(args.output, "w") as fp:
        for s in ranked_genes:
            fp.write("%s\n" % s)

    print("Done!")

if __name__ == "__main__":
    get_features()

import numpy as np

import openpyxl
import json
import os

import argparse


def main(param):
    print(f"Received parameter: {param}")



def export_all_scores_excel(score_dir):


    all_scores_files = os.listdir(score_dir)
    all_scores_files.sort()

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(['scene', 'abs_diff', 'abs_rel', 'sq_rel', 'rmse', 'rmse_log', 'a5', 'a10', 'a25', 'a0', 'a1', 'a2', 'a3', 'model_time'])


    for score_file in all_scores_files:
        if score_file.endswith(".json") and "_00_" in score_file:
            with open(os.path.join(score_dir, score_file)) as f:
                data = json.load(f)
                name = data['metrics_type'].split(" ")[1][5:9]
                score = [float(s) for s in data['scores_string'].split(',') if s.strip()]
                ws.append([name, *score])
                # break


    wb.save(os.path.join(score_dir, "all_scores.xlsx"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a parameter.")
    parser.add_argument("score_dir", type=str, help="A parameter to process",required=True)
    #    score_dir = "/data/laiyan/codes/simplerecon/results/HERO_MODEL/scannet/default/scores/"
    args = parser.parse_args()
    main(args.score_dir)
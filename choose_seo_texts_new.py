import glob
from collections import defaultdict
import pandas as pd
from utils import run_bash_command
from tqdm import tqdm
import os
import utils


def read_working_set_file(working_set_file_path):
    init_df = pd.read_csv(working_set_file_path, delimiter=' ', header=None)
    init_df = init_df.rename({0: "qID", 2: "docno"}, axis=1)[["qID", "docno"]]
    init_docs = init_df.groupby("qID").docno.apply(list).to_dict()
    return init_docs


def read_feature_files(features_dir, init_docs, feature_list, stream_list):
    features = defaultdict(lambda: defaultdict(dict))
    feature_id = {}
    no_stream = defaultdict(dict)
    f_id = 0
    feature_id_path = os.path.join(features_dir, 'featureID')
    lines = []

    for feature in tqdm(feature_list):
        for stream in stream_list:
            feature_name = f"{stream}{feature}"
            f_id += 1
            feature_id[feature_name] = f_id
            lines.append(f"{feature_name} {f_id}\n")

            for qID in init_docs:
                feature_file = os.path.join(features_dir, f"doc_{feature}_{qID}")
                if os.path.exists(feature_file):
                    with open(feature_file, 'r') as fi:
                        for line in fi:
                            dID, score = line.strip().split()
                            dID = dID.split("$")[-1]
                            if dID in init_docs[qID]:
                                features[feature_name][qID][dID] = float(score)
                else:
                    no_stream[feature_name][qID] = 1

    with open(feature_id_path, 'w') as fo:
        fo.writelines(lines)

    return features, feature_id, no_stream


def normalize_features(features):
    min_max_values = {}
    for feature_name, queries in features.items():
        all_scores = [score for query in queries.values() for score in query.values()]
        min_max_values[feature_name] = (min(all_scores), max(all_scores))
    normalized_features = defaultdict(lambda: defaultdict(dict))
    for feature_name, queries in features.items():
        min_value, max_value = min_max_values[feature_name]
        range_value = max_value - min_value
        for qID, docs in queries.items():
            for dID, score in docs.items():
                if range_value == 0:
                    normalized_score = 0
                else:
                    normalized_score = (score - min_value) / range_value
                normalized_features[feature_name][qID][dID] = normalized_score
    return normalized_features


def generate_output_file_matching_perl(output_file_path, normalized_features, feature_id, init_docs):
    lines = []
    for qID in sorted(init_docs.keys(), key=int):
        for dID in init_docs[qID]:
            row = []
            qID_ = qID.replace("_", "")
            row.append(f"0 qid:{qID_}")
            for featureName in sorted(feature_id, key=lambda x: feature_id[x]):
                res = 0.0
                if (featureName in normalized_features and qID in normalized_features[featureName] and dID in
                        normalized_features[featureName][qID]):
                    res = normalized_features[featureName][qID][dID]
                res = f"{res:.8f}"
                row.append(f" {feature_id[featureName]}:{res}")
            row.append(f" # {dID}\n")

            line = ''.join(row)
            lines.append(''.join(row))

    with open(output_file_path, 'w') as file:
        file.writelines(lines)


if __name__ == '__main__':

    g_df = pd.read_csv("/lv_local/home/user/CharPDM/g_data.csv")
    g_df = g_df[g_df.round_no == 7]

    archive_dir = "archive_test_w2v"
    model_dir = "/lv_local/home/user/train_RankSVM/feature_sets"

    features_dir = '/lv_local/home/user/content_modification_code-master/g_output/output_feature_files_dir'
    feature_list = ["FractionOfQueryWordsIn", "FractionOfQueryWordsOut", "CosineToCentroidIn", "CosineToCentroidInVec",
                    "CosineToCentroidOut", "CosineToCentroidOutVec", "CosineToWinnerCentroidInVec",
                    "CosineToWinnerCentroidOutVec", "CosineToWinnerCentroidIn", "CosineToWinnerCentroidOut",
                    "SimilarityToPrev", "SimilarityToRefSentence", "SimilarityToPred", "SimilarityToPrevRef",
                    "SimilarityToPredRef"]
    stream_list = ["test_"]


    file_to_nick = {k:k.split("_")[-1] for k in os.listdir(model_dir) if "model" in k}

    for model in file_to_nick.keys():
        print(f"\n\n########## model: {model} ##########\n\n")

        nick = file_to_nick[model]
        for pos in tqdm(["2", "3", "4", "5"]):

            working_set_file_path = f'/lv_local/home/user/content_modification_code-master/g_output/saved_result_files/{archive_dir}/ws_output_{pos}.txt'

            command = f"/lv_local/home/user/svm_rank/svm_rank_classify " \
                      f"/lv_local/home/user/content_modification_code-master/g_output/saved_result_files/{archive_dir}/features_{pos}.dat " \
                      f"{model_dir}/{model} " \
                      f"/lv_local/home/user/content_modification_code-master/g_output/saved_result_files/predictions_{nick}_{pos}.txt"
            out = run_bash_command(command)

            text_df = pd.DataFrame([line.strip().split(None, 2) for line in
                                    open(f'./g_output/saved_result_files/{archive_dir}/raw_ds_out_{pos}_texts.txt')],
                                   columns=['index_', 'ID', 'text'])
            text_df[['ref', 'docid']] = text_df['ID'].str.split('$', n=1, expand=True)
            text_df["creator"] = text_df["ref"].str.split("-", expand=True)[3].astype(int)
            text_df["query_id"] = text_df["ref"].str.split("-", expand=True)[2].astype(int)

            df = pd.read_csv(working_set_file_path, delimiter=' ', header=None).sort_values([0, 2])
            df["score"] = pd.read_csv(
                f"/lv_local/home/user/content_modification_code-master/g_output/saved_result_files/predictions_{nick}_{pos}.txt",
                header=None)
            df['rank'] = df.groupby(0)['score'].rank(method='first', ascending=False).astype(int)
            df = df.rename(columns={0: 'qid', 2: 'docid'})[["qid", "docid", "score", "rank"]].sort_values(
                ['qid', 'docid'])
            df_rank1 = df.query('rank == 1')
            df_rank1 = df_rank1[df_rank1.docid.str.contains("ROUND-07")]  # test data according to the article
            df_rank1["round_no"] = "07"
            final_df = pd.merge(df_rank1, text_df, on='docid', how='left')
            final_df["username"] = "BOT_" + nick
            final_df = final_df[["round_no", "query_id", "creator", "username", "text"]]
            final_df["round_no"] = final_df["round_no"].str.replace("0", "")

            final_df.to_csv(f"./g_output/saved_result_files/bot_followup_comp_{nick}_{pos}.csv", index=False)

    df = pd.concat(
        [pd.read_csv(file) for file in glob.glob("./g_output/saved_result_files/bot_followup_comp_*.csv")],
        ignore_index=True).sort_values(["round_no", "query_id", "creator"])

    df.to_csv("./g_output/saved_result_files/bot_followup_comp.csv", index=False)
    print("\n\n########## created bot followup file for all models and positions ##########\n\n")

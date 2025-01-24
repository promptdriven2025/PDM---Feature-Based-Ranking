import glob
from collections import defaultdict
import pandas as pd
from utils import run_bash_command
from tqdm import tqdm
import os
import utils
import warnings

warnings.filterwarnings("ignore")

if __name__ == '__main__':


    g_df = pd.read_csv("/lv_local/home/user/CharPDM/t_data.csv") 
    g_df = g_df[g_df.round_no == 4]

    archive_dir = "archive_test_w2v"
    model_dir = "/lv_local/home/user/train_RankSVM/feature_sets"

    features_dir = '/lv_local/home/user/content_modification_code-master/g_output/output_feature_files_dir'
    feature_list = ["FractionOfQueryWordsIn", "FractionOfQueryWordsOut", "CosineToCentroidIn", "CosineToCentroidInVec",
                    "CosineToCentroidOut", "CosineToCentroidOutVec", "CosineToWinnerCentroidInVec",
                    "CosineToWinnerCentroidOutVec", "CosineToWinnerCentroidIn", "CosineToWinnerCentroidOut",
                    "SimilarityToPrev", "SimilarityToRefSentence", "SimilarityToPred", "SimilarityToPrevRef",
                    "SimilarityToPredRef"]
    stream_list = ["test_"]

    file_to_nick = {k: k.split("_")[-1] for k in os.listdir(model_dir) if "model" in k}
    
    print(f"\n\n########## model number: {len(file_to_nick.keys())} ##########\n\n")

    for model in file_to_nick.keys():
        print(f"\n\n########## model: {model} ##########\n\n")

        nick = file_to_nick[model]
        if "$" in model:
            print("ERROR! bot name contains $")

        for pos in tqdm(["2", "3", "4", "5"]):
            if os.path.exists(f"./g_output/saved_result_files/bot_followup_asrc_{nick}_{pos}.csv"):
                continue

            working_set_file_path = f'/lv_local/home/user/content_modification_code-master/g_output/saved_result_files/{archive_dir}/ws_output_{pos}.txt'

            features_file_path = f'/lv_local/home/user/content_modification_code-master/g_output/saved_result_files/{archive_dir}/features_{pos}.dat'
            predictions_file_path = f'/lv_local/home/user/content_modification_code-master/g_output/saved_result_files/predictions_{nick}_{pos}.txt'

            command = f"/lv_local/home/user/opt/java/jdk1.8.0/bin/java -jar /lv_local/home/user/content_modification_code-master/scripts/RankLib.jar -load {model_dir}/{model} -rank {features_file_path} -score {predictions_file_path}"
            
            out = run_bash_command(command)

            if "Exception" in str(out):
                utils.check_and_update_permissions("/lv_local/home/user/opt/java/jdk1.8.0/bin/java")
                utils.check_and_update_permissions(
                    "/lv_local/home/user/content_modification_code-master/scripts/RankLib.jar")
                utils.check_and_update_permissions(f"{model_dir}/{model}")
                utils.check_and_update_permissions(features_file_path)
                out = run_bash_command(command)
                if "Exception" in str(out):
                    raise Exception(str(out))
                x = 1

            print(out)
            text_df = pd.DataFrame([line.strip().split(None, 2) for line in
                                    open(f'./g_output/saved_result_files/{archive_dir}/raw_ds_out_{pos}_texts.txt')],
                                   columns=['index_', 'ID', 'text'])
            text_df[['ref', 'docid']] = text_df['ID'].str.split('$', n=1, expand=True)
            text_df["creator"] = text_df["ref"].str.split("-", expand=True)[3].astype(int)
            text_df["query_id"] = text_df["ref"].str.split("-", expand=True)[2].astype(int)

            df = pd.read_csv(working_set_file_path, delimiter=' ', header=None).sort_values([0, 2])
            score_column = pd.read_csv(predictions_file_path, header=None, delimiter='\t', usecols=[2])

            
            if int(score_column.isna().sum()) > 0:
                print(f"ERROR in {model}_{pos}: score column contains NaN values")
                continue

            df["score"] = score_column
            df['rank'] = df.groupby(0)['score'].rank(method='first', ascending=False).astype(int)
            df = df.rename(columns={0: 'qid', 2: 'docid'})[["qid", "docid", "score", "rank"]].sort_values(
                ['qid', 'docid'])
            df_rank1 = df.query('rank == 1')
            df_rank1 = df_rank1[df_rank1.docid.str.contains("ROUND-07")]  
            df_rank1["round_no"] = "07"

            final_df = pd.merge(df_rank1, text_df, on='docid', how='left')
            final_df["username"] = "BOT_" + nick
            final_df = final_df[["round_no", "query_id", "creator", "username", "text"]]
            final_df["round_no"] = final_df["round_no"].str.replace("0", "")

            final_df.to_csv(f"./g_output/saved_result_files/bot_followup_asrc_{nick}_{pos}.csv", index=False)


    df = pd.concat(
        [pd.read_csv(file) for file in glob.glob("./g_output/saved_result_files/bot_followup_asrc_*.csv")],
        ignore_index=True).sort_values(["round_no", "query_id", "creator"])

    df.to_csv("./g_output/saved_result_files/bot_followup_asrc.csv", index=False)
    print("\n\n########## created bot followup file for all models and positions ##########\n\n")

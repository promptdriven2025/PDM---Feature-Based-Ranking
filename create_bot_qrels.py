import pandas as pd


def create_bot_followp():
    dir_path = '/lv_local/home/user/content_modification_code-master/g_output/saved_result_files/archive_train_w2v'
    col1 = []
    col2 = []
    col3 = []
    for i in ['2', '5']:
        with open(f'{dir_path}/raw_ds_out_{i}_texts.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.split(maxsplit=2) 
                col1.append(parts[0])
                col2.append(parts[1])
                col3.append(parts[2].strip())
    df = pd.DataFrame({'query_id_full': col1, 'docno': col2, 'text': col3})
    df['query_id'], df['round_no'] = df.query_id_full.apply(lambda x: x[:-2]), df.query_id_full.apply(lambda x: x[-2:])
    df = df[df.round_no == '06']
    df['round_no'] = '6'
    df['creator'] = df.docno.apply(lambda x: x.split('$')[0].split('-')[-1])
    df['username'] = df.docno
    df = df[['round_no', 'query_id', 'creator', 'username', 'text']]
    df.to_csv(f'/lv_local/home/user/content_modification_code-master/g_output/bot_followup_compqrels.csv',
              index=False)


def create_bot_followp_test():
    dir_path = '/lv_local/home/user/content_modification_code-master/g_output/saved_result_files/archive_test_w2v'
    col1 = []
    col2 = []
    col3 = []
    for i in ['2', '3', '4', '5']:
        with open(f'{dir_path}/raw_ds_out_{i}_texts.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.split(maxsplit=2)  # Split only on the first two spaces
                col1.append(parts[0])
                col2.append(parts[1])
                col3.append(parts[2].strip())
    df = pd.DataFrame({'query_id_full': col1, 'docno': col2, 'text': col3})
    df['query_id'], df['round_no'] = df.query_id_full.apply(lambda x: x[:-2]), df.query_id_full.apply(lambda x: x[-2:])
    df = df[df.round_no == '07']
    df['round_no'] = '7'
    df['creator'] = df.docno.apply(lambda x: x.split('$')[0].split('-')[-1])
    df['username'] = df.docno
    df = df[['round_no', 'query_id', 'creator', 'username', 'text']]
    df.to_csv(f'/lv_local/home/user/content_modification_code-master/g_output/bot_followup_compqrelstest.csv',
              index=False)


def get_qrels_file():
    file_path = f'/lv_local/home/user/content_modification_code-master/Results/RankedLists/LambdaMARTcompqrels'
    columns = ['query_id_long', 'Q0', 'docno_long', 'rank', 'score', 'method']

    with open(file_path, 'r') as file:
        lines = file.readlines()

    data = []

    for line in lines:
        values = line.strip().split()
        query_id, _, document_id, rank, score, method = values
        query_id_padded = "{:03}".format(int(query_id))
        data.append([query_id_padded, 'Q0', document_id, int(rank), float(score), method])

    df = pd.DataFrame(data, columns=columns).drop(["Q0", "method", "score"], axis=1)
    df['docno_long'] = df['docno_long'].apply(lambda x: '-'.join(x.split('-')[2:-1]))
    df = df[df.docno_long.str.contains('ROUND')]
    df["docno"] = df["docno_long"].apply(lambda x: x.split('$')[0])
    df["query_id"] = df["query_id_long"].apply(lambda x: int(x[1:4]))
    g_df = pd.read_csv("/lv_local/home/user/CharPDM/g_data.csv")[['docno', 'round_no', 'query_id', 'position']]
    g_df = g_df[g_df.round_no == 6]
    merged_df = pd.merge(df, g_df, on=['docno', 'query_id'], how='left')
    merged_df['rank_promotion'] = merged_df['position'] - merged_df['rank']
    merged_df['rank_promotion'] = merged_df['rank_promotion'].apply(lambda x: 0 if x < 0 else x)
    merged_df["final_qid"] = merged_df.apply(
        lambda row: row.query_id_long[1:4] + '6' + (str(row.position) if row.position != 6 else '5'), axis=1)
    merged_df["final_docno"] = merged_df.docno_long.apply(lambda x: x.split("$")[1])
    merged_df = merged_df[['final_qid', 'final_docno', 'rank_promotion']]

    lines = []
    for idx, row in merged_df.iterrows():
        lines.append(f"{row.final_qid} 0 {row.final_docno} {row.rank_promotion}\n")

    with open("/lv_local/home/user/train_RankSVM/qrels_n.txt", 'w') as file:
        file.writelines(lines)


if __name__ == '__main__':
    create_bot_followp_test()

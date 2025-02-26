import pandas as pd

def process_csv(input_file, output_file, num_celebrity=100):
    # CSVファイルを読み込む
    df = pd.read_csv(input_file)

    # artist_idカラムを初期化
    df['concept_id'] = -1

    count_erased_rows = df[df['type'] == 'erased'].shape[0]
    print(f"Number of erased rows: {count_erased_rows}")
    num_rows_per_celebrity = count_erased_rows // num_celebrity

    # 各行に対して処理を行う
    for index, row in df.iterrows():
        if row['type'] =='erased':
            df.at[index, 'concept_id'] = index // num_rows_per_celebrity

    # 結果を出力
    print(df)
    df.to_csv(output_file, index=False)


process_csv('prompts_csv/celebrity_1_concepts.csv', 'prompts_csv/celebrity_1_concepts2.csv', 1)
process_csv('prompts_csv/celebrity_5_concepts.csv', 'prompts_csv/celebrity_5_concepts2.csv', 5)
process_csv('prompts_csv/celebrity_10_concepts.csv', 'prompts_csv/celebrity_10_concepts2.csv', 10)
process_csv('prompts_csv/celebrity_100_concepts.csv', 'prompts_csv/celebrity_100_concepts2.csv', 100)
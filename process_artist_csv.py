import pandas as pd

# CSVファイルを読み込む
df = pd.read_csv('prompts_csv/art_100_concepts.csv')

# artist_idカラムを初期化
df['concept_id'] = -1

# artist_idを割り当てるための変数
current_id = 0
seen_artists = set()

# 各行に対して処理を行う
for index, row in df.iterrows():
    if row['type'] == 0:
        # typeが0の場合、artistが新しいかチェック
        artist = row['artist']
        if artist not in seen_artists:
            # 新しいartistの場合、新しいIDを割り当て
            seen_artists.add(artist)
            df.at[index, 'concept_id'] = current_id
            current_id += 1
        else:
            # 既に見たartistの場合、そのartistの最初のIDを使用
            artist_id = df[(df['artist'] == artist) & (df['concept_id'] != -1)].iloc[0]['concept_id']
            df.at[index, 'concept_id'] = artist_id
    # typeが0以外の場合は-1のまま

# 結果を出力
print(df)
df.to_csv('prompts_csv/art_100_concepts2.csv', index=False)

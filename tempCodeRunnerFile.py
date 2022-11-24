mecab = Mecab()
# train_data['tokenized'] = train_data['article'].apply(mecab.morphs)
# train_data['tokenized'] = train_data['tokenized'].apply(
#     lambda x: [item for item in x if item not in stopwords])
# test_data['tokenized'] = test_data['article'].apply(mecab.morphs)
# test_data['tokenized'] = test_data['tokenized'].apply(
#     lambda x: [item for item in x if item not in stopwords])

# negative_words = np.hstack(
#     train_data[train_data.label == -1]['tokenized'].values)
# positive_words = np.hstack(
#     train_data[train_data.label == 1]['tokenized'].values)

# negative_word_count = Counter(negative_words)
# print(negative_word_count.most_common(20))
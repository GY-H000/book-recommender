import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import pickle

print("加载数据...")
ratings = pd.read_csv('Ratings.csv')
ratings = ratings[ratings['Book-Rating'] > 0]

# 过滤：至少20个评分的书，至少10个评分的用户
book_counts = ratings.groupby('ISBN')['Book-Rating'].count()
user_counts = ratings.groupby('User-ID')['Book-Rating'].count()
ratings = ratings[ratings['ISBN'].isin(book_counts[book_counts >= 20].index)]
ratings = ratings[ratings['User-ID'].isin(user_counts[user_counts >= 10].index)]

# 编码用户和书籍ID
user_ids = ratings['User-ID'].unique()
book_ids = ratings['ISBN'].unique()
user_to_idx = {uid: i for i, uid in enumerate(user_ids)}
book_to_idx = {bid: i for i, bid in enumerate(book_ids)}
idx_to_book = {i: bid for bid, i in book_to_idx.items()}

ratings['user_idx'] = ratings['User-ID'].map(user_to_idx)
ratings['book_idx'] = ratings['ISBN'].map(book_to_idx)

# 归一化评分到0-1
ratings['rating_norm'] = ratings['Book-Rating'] / 10.0

X = ratings[['user_idx', 'book_idx']].values
y = ratings['rating_norm'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"用户数: {len(user_ids)}, 书籍数: {len(book_ids)}, 评分数: {len(ratings)}")

# 构建神经协同过滤模型
user_input = keras.Input(shape=(1,), name='user')
book_input = keras.Input(shape=(1,), name='book')

user_embedding = keras.layers.Embedding(len(user_ids), 50, name='user_emb')(user_input)
book_embedding = keras.layers.Embedding(len(book_ids), 50, name='book_emb')(book_input)

user_vec = keras.layers.Flatten()(user_embedding)
book_vec = keras.layers.Flatten()(book_embedding)

concat = keras.layers.Concatenate()([user_vec, book_vec])
dense1 = keras.layers.Dense(128, activation='relu')(concat)
dropout1 = keras.layers.Dropout(0.3)(dense1)
dense2 = keras.layers.Dense(64, activation='relu')(dropout1)
dropout2 = keras.layers.Dropout(0.3)(dense2)
output = keras.layers.Dense(1, activation='sigmoid')(dropout2)

model = keras.Model(inputs=[user_input, book_input], outputs=output)
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

print("\n开始训练...")
history = model.fit(
    [X_train[:, 0], X_train[:, 1]], y_train,
    validation_data=([X_test[:, 0], X_test[:, 1]], y_test),
    epochs=10,
    batch_size=256,
    verbose=1
)

print("\n保存模型...")
model.save('book_model.keras')
with open('mappings.pkl', 'wb') as f:
    pickle.dump({
        'user_to_idx': user_to_idx,
        'book_to_idx': book_to_idx,
        'idx_to_book': idx_to_book
    }, f)

print("✅ 训练完成！")

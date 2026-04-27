import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pickle

print("加载数据...")
ratings = pd.read_csv('Ratings.csv')
ratings = ratings[ratings['Book-Rating'] > 0]

book_counts = ratings.groupby('ISBN')['Book-Rating'].count()
user_counts = ratings.groupby('User-ID')['Book-Rating'].count()
ratings = ratings[ratings['ISBN'].isin(book_counts[book_counts >= 20].index)]
ratings = ratings[ratings['User-ID'].isin(user_counts[user_counts >= 10].index)]

user_ids = ratings['User-ID'].unique()
book_ids = ratings['ISBN'].unique()
user_to_idx = {uid: i for i, uid in enumerate(user_ids)}
book_to_idx = {bid: i for i, bid in enumerate(book_ids)}
idx_to_book = {i: bid for bid, i in book_to_idx.items()}

ratings['user_idx'] = ratings['User-ID'].map(user_to_idx)
ratings['book_idx'] = ratings['ISBN'].map(book_to_idx)
ratings['rating_norm'] = ratings['Book-Rating'] / 10.0

X = ratings[['user_idx', 'book_idx']].values
y = ratings['rating_norm'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"用户数: {len(user_ids)}, 书籍数: {len(book_ids)}, 评分数: {len(ratings)}")


class RatingDataset(Dataset):
    def __init__(self, X, y):
        self.users = torch.LongTensor(X[:, 0])
        self.books = torch.LongTensor(X[:, 1])
        self.ratings = torch.FloatTensor(y)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.users[idx], self.books[idx], self.ratings[idx]


class NCF(nn.Module):
    def __init__(self, n_users, n_books, emb_dim=50):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.book_emb = nn.Embedding(n_books, emb_dim)
        self.fc = nn.Sequential(
            nn.Linear(emb_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, user, book):
        x = torch.cat([self.user_emb(user), self.book_emb(book)], dim=1)
        return self.fc(x).squeeze()


train_loader = DataLoader(RatingDataset(X_train, y_train), batch_size=256, shuffle=True)
test_loader = DataLoader(RatingDataset(X_test, y_test), batch_size=256)

model = NCF(len(user_ids), len(book_ids))
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

print("\n开始训练...")
for epoch in range(10):
    model.train()
    train_loss = sum(
        (lambda loss: (optimizer.zero_grad(), loss.backward(), optimizer.step(), loss.item())[-1])(
            criterion(model(u, b), r)
        )
        for u, b, r in train_loader
    ) / len(train_loader)

    model.eval()
    with torch.no_grad():
        val_loss = sum(criterion(model(u, b), r).item() for u, b, r in test_loader) / len(test_loader)

    print(f"Epoch {epoch+1}/10 - loss: {train_loss:.4f} - val_loss: {val_loss:.4f}")

print("\n保存模型...")
torch.save(model.state_dict(), 'book_model.pt')
with open('mappings.pkl', 'wb') as f:
    pickle.dump({
        'user_to_idx': user_to_idx,
        'book_to_idx': book_to_idx,
        'idx_to_book': idx_to_book,
        'n_users': len(user_ids),
        'n_books': len(book_ids)
    }, f)

print("✅ 训练完成！运行 streamlit run app.py 启动推荐系统")

import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import pickle


class NCF(nn.Module):
    def __init__(self, n_users, n_books, emb_dim=50):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.book_emb = nn.Embedding(n_books, emb_dim)
        self.fc = nn.Sequential(
            nn.Linear(emb_dim * 2, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 1), nn.Sigmoid()
        )

    def forward(self, user, book):
        x = torch.cat([self.user_emb(user), self.book_emb(book)], dim=1)
        return self.fc(x).squeeze()

st.set_page_config(page_title="书荒救星", page_icon="📚", layout="wide")

st.markdown("""
<style>
    .main {background-color: #f5f7fa;}
    .stButton>button {background-color: #4CAF50; color: white; border-radius: 8px;}
    .book-card {
        background: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

st.title("📚 书荒救星 - 智能图书推荐")
st.markdown("### 🎯 根据你的口味，找到下一本好书")

@st.cache_data
def load_data():
    books = pd.read_csv('Books.csv', low_memory=False)
    ratings = pd.read_csv('Ratings.csv')
    books_clean = books.dropna(subset=['Book-Author', 'Publisher'])
    ratings_clean = ratings[ratings['Book-Rating'] > 0]

    # 计算平均评分
    avg_ratings = ratings_clean.groupby('ISBN').agg({
        'Book-Rating': ['mean', 'count']
    }).reset_index()
    avg_ratings.columns = ['ISBN', 'avg_rating', 'rating_count']
    books_clean = books_clean.merge(avg_ratings, on='ISBN', how='left')

    # 协同过滤
    book_counts = ratings_clean.groupby('ISBN')['Book-Rating'].count()
    popular_isbns = book_counts[book_counts >= 20].index
    ratings_popular = ratings_clean[ratings_clean['ISBN'].isin(popular_isbns)]
    user_counts = ratings_popular.groupby('User-ID')['Book-Rating'].count()
    active_users = user_counts[user_counts >= 10].index
    ratings_filtered = ratings_popular[ratings_popular['User-ID'].isin(active_users)]
    book_matrix = ratings_filtered.pivot_table(
        index='User-ID', columns='ISBN', values='Book-Rating'
    ).fillna(0)
    book_similarity = cosine_similarity(book_matrix.T)
    book_sim_df = pd.DataFrame(
        book_similarity,
        index=book_matrix.columns,
        columns=book_matrix.columns
    )

    # 基于内容的相似度（作者+出版社）
    books_clean['content'] = books_clean['Book-Author'].fillna('') + ' ' + books_clean['Publisher'].fillna('')
    tfidf = TfidfVectorizer(max_features=500)
    content_matrix = tfidf.fit_transform(books_clean['content'])
    content_sim = cosine_similarity(content_matrix)

    return books_clean, book_sim_df, content_sim, ratings_clean

@st.cache_resource
def load_dl_model():
    if os.path.exists('book_model.pt') and os.path.exists('mappings.pkl'):
        with open('mappings.pkl', 'rb') as f:
            mappings = pickle.load(f)
        model = NCF(mappings['n_users'], mappings['n_books'])
        model.load_state_dict(torch.load('book_model.pt', map_location='cpu'))
        model.eval()
        return model, mappings
    return None, None

books_clean, book_sim_df, content_sim, ratings_clean = load_data()
dl_model, mappings = load_dl_model()

# 侧边栏 - 个性化选项
with st.sidebar:
    st.header("🎨 个性化设置")

    methods = ["混合推荐（推荐）", "基于用户评分", "基于作者风格"]
    if dl_model is not None:
        methods.insert(1, "🤖 深度学习推荐")
        st.success("✅ 深度学习模型已加载")
    else:
        st.info("💡 运行 train_model.py 启用深度学习")

    rec_method = st.radio("推荐方式", methods, help="混合推荐结合多种算法，效果最佳")
    num_recs = st.slider("推荐数量", 3, 10, 5)
    min_rating = st.slider("最低评分", 0.0, 10.0, 6.0, 0.5)

# 主搜索区
col1, col2 = st.columns([3, 1])
with col1:
    keyword = st.text_input(
        "🔍 搜索书籍",
        placeholder="输入书名、作者或关键词（中英文均可）",
        help="支持模糊搜索，例如：哈利、Potter、魔法"
    )
with col2:
    st.write("")
    st.write("")
    search_in = st.selectbox("搜索范围", ["书名", "作者", "全部"])

# 中英文关键词映射
zh_map = {
    "魔法": "magic", "哈利": "harry", "波特": "potter", "魔戒": "ring",
    "指环王": "lord ring", "小王子": "little prince", "夏洛": "charlotte",
    "纳尼亚": "narnia", "福尔摩斯": "holmes sherlock", "侦探": "detective mystery",
    "爱情": "love romance", "战争": "war", "历史": "history", "科幻": "science fiction",
    "奇幻": "fantasy", "悬疑": "mystery thriller", "恐怖": "horror", "传记": "biography"
}

if keyword:
    search_word = keyword.lower()
    for zh, en in zh_map.items():
        if zh in keyword:
            search_word += " " + en

    # 根据搜索范围匹配
    if search_in == "书名":
        matched = books_clean[books_clean['Book-Title'].str.contains(search_word, case=False, na=False)]
    elif search_in == "作者":
        matched = books_clean[books_clean['Book-Author'].str.contains(search_word, case=False, na=False)]
    else:
        matched = books_clean[
            books_clean['Book-Title'].str.contains(search_word, case=False, na=False) |
            books_clean['Book-Author'].str.contains(search_word, case=False, na=False)
        ]

    if matched.empty:
        st.error("😢 没有找到相关书籍，试试其他关键词或更换搜索范围")
    else:
        st.success(f"找到 {len(matched)} 本相关书籍")
        options = matched.nlargest(20, 'rating_count', keep='first')['Book-Title'].tolist()
        selected = st.selectbox("📖 选择一本你喜欢的书：", options)

        if selected:
            book_row = matched[matched['Book-Title'] == selected].iloc[0]
            isbn = book_row['ISBN']

            # 显示选中的书籍信息
            st.markdown("---")
            col1, col2 = st.columns([1, 3])
            with col1:
                if pd.notna(book_row['Image-URL-M']):
                    st.image(book_row['Image-URL-M'], width=150)
            with col2:
                st.markdown(f"### {book_row['Book-Title']}")
                st.write(f"**作者:** {book_row['Book-Author']}")
                st.write(f"**出版社:** {book_row['Publisher']}")
                if pd.notna(book_row['avg_rating']):
                    st.write(f"**平均评分:** ⭐ {book_row['avg_rating']:.1f}/10 ({int(book_row['rating_count'])} 人评价)")

            st.markdown("---")
            st.markdown(f"### 🎁 因为你喜欢《{selected}》，为你推荐：")

            # 推荐逻辑
            recommendations = []

            if rec_method == "🤖 深度学习推荐" and dl_model is not None:
                if isbn not in mappings['book_to_idx']:
                    st.warning("😅 该书不在深度学习训练集中，已切换为混合推荐")
                    rec_method = "混合推荐（推荐）"
                else:
                    raters = ratings_clean[
                        (ratings_clean['ISBN'] == isbn) & (ratings_clean['Book-Rating'] >= 7)
                    ]['User-ID'].values
                    valid_users = [u for u in raters[:50] if u in mappings['user_to_idx']]
                    candidate_isbns = [b for b in list(book_sim_df.columns[:500]) if b != isbn and b in mappings['book_to_idx']]

                    if valid_users and candidate_isbns:
                        u_idxs, b_idxs, b_isbns = [], [], []
                        for b_isbn in candidate_isbns:
                            b_idx = mappings['book_to_idx'][b_isbn]
                            for u in valid_users:
                                u_idxs.append(mappings['user_to_idx'][u])
                                b_idxs.append(b_idx)
                                b_isbns.append(b_isbn)

                        with torch.no_grad():
                            preds = dl_model(torch.LongTensor(u_idxs), torch.LongTensor(b_idxs)).numpy()

                        scores = pd.DataFrame({'isbn': b_isbns, 'score': preds}) \
                            .groupby('isbn')['score'].mean().sort_values(ascending=False)
                        recommendations = books_clean[books_clean['ISBN'].isin(scores.head(num_recs * 2).index)]
                    else:
                        st.warning("😅 没有足够的用户数据，已切换为混合推荐")
                        rec_method = "混合推荐（推荐）"

            if rec_method == "基于用户评分" and isbn in book_sim_df.columns:
                similar = book_sim_df[isbn].sort_values(ascending=False)[1:num_recs+1]
                recommendations = books_clean[books_clean['ISBN'].isin(similar.index)]
            elif rec_method == "基于作者风格":
                book_idx = books_clean[books_clean['ISBN'] == isbn].index[0]
                similar_idx = np.argsort(content_sim[book_idx])[::-1][1:num_recs+1]
                recommendations = books_clean.iloc[similar_idx]
            elif rec_method not in ("🤖 深度学习推荐",):  # 混合推荐
                if isbn in book_sim_df.columns:
                    cf_similar = book_sim_df[isbn].sort_values(ascending=False)[1:num_recs*2]
                    cf_books = books_clean[books_clean['ISBN'].isin(cf_similar.index)]
                else:
                    cf_books = pd.DataFrame()

                book_idx = books_clean[books_clean['ISBN'] == isbn].index[0]
                similar_idx = np.argsort(content_sim[book_idx])[::-1][1:num_recs*2]
                content_books = books_clean.iloc[similar_idx]

                recommendations = pd.concat([cf_books, content_books]).drop_duplicates(subset='ISBN').head(num_recs)

            # 过滤评分
            if not recommendations.empty:
                recommendations = recommendations[
                    (recommendations['avg_rating'] >= min_rating) |
                    (recommendations['avg_rating'].isna())
                ]

            if recommendations.empty:
                st.warning("😅 没有找到符合条件的推荐，试试降低最低评分要求")
            else:
                for idx, row in recommendations.iterrows():
                    col1, col2 = st.columns([1, 4])
                    with col1:
                        if pd.notna(row['Image-URL-M']):
                            st.image(row['Image-URL-M'], width=100)
                    with col2:
                        st.markdown(f"#### 📖 {row['Book-Title']}")
                        st.write(f"**作者:** {row['Book-Author']}")
                        if pd.notna(row['avg_rating']):
                            st.write(f"⭐ {row['avg_rating']:.1f}/10 ({int(row['rating_count'])} 人评价)")
                    st.markdown("---")
else:
    st.info("💡 输入书名或作者开始搜索，系统会为你推荐相似的好书！")

    # 显示热门书籍
    st.markdown("### 🔥 热门书籍")
    popular = books_clean.nlargest(6, 'rating_count', keep='first')
    cols = st.columns(3)
    for i, (_, row) in enumerate(popular.iterrows()):
        with cols[i % 3]:
            if pd.notna(row['Image-URL-M']):
                st.image(row['Image-URL-M'], width=150)
            st.write(f"**{row['Book-Title'][:30]}...**" if len(row['Book-Title']) > 30 else f"**{row['Book-Title']}**")
            st.write(f"👤 {row['Book-Author']}")
            if pd.notna(row['avg_rating']):
                st.write(f"⭐ {row['avg_rating']:.1f}")
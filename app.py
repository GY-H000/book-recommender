import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import pickle
import requests


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
st.markdown("### 🎯 中英文书籍一站搜索，找到下一本好书")


# ── 数据加载 ──────────────────────────────────────────────────────────────────

@st.cache_data
def load_data():
    books = pd.read_csv('Books.csv', low_memory=False)
    ratings = pd.read_csv('Ratings.csv')
    books_clean = books.dropna(subset=['Book-Author', 'Publisher'])
    ratings_clean = ratings[ratings['Book-Rating'] > 0]

    avg_ratings = ratings_clean.groupby('ISBN').agg(
        avg_rating=('Book-Rating', 'mean'),
        rating_count=('Book-Rating', 'count')
    ).reset_index()
    books_clean = books_clean.merge(avg_ratings, on='ISBN', how='left')

    # 协同过滤矩阵
    book_counts = ratings_clean.groupby('ISBN')['Book-Rating'].count()
    popular_isbns = book_counts[book_counts >= 20].index
    ratings_popular = ratings_clean[ratings_clean['ISBN'].isin(popular_isbns)]
    user_counts = ratings_popular.groupby('User-ID')['Book-Rating'].count()
    active_users = user_counts[user_counts >= 10].index
    ratings_filtered = ratings_popular[ratings_popular['User-ID'].isin(active_users)]
    book_matrix = ratings_filtered.pivot_table(
        index='User-ID', columns='ISBN', values='Book-Rating'
    ).fillna(0)
    book_sim_df = pd.DataFrame(
        cosine_similarity(book_matrix.T),
        index=book_matrix.columns,
        columns=book_matrix.columns
    )

    # 内容相似度（top-K 邻居，避免 OOM）
    content_books = books_clean[books_clean['rating_count'] >= 5].reset_index(drop=True)
    content_books['content'] = (
        content_books['Book-Author'].fillna('') + ' ' +
        content_books['Publisher'].fillna('')
    )
    tfidf = TfidfVectorizer(max_features=500)
    content_matrix = tfidf.fit_transform(content_books['content'])

    K = 50
    isbn_arr = content_books['ISBN'].values
    top_k_neighbors = {}
    batch_size = 500
    n = len(content_books)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        sims = cosine_similarity(content_matrix[start:end], content_matrix)
        for i, row_sims in enumerate(sims):
            global_idx = start + i
            top_idx = np.argpartition(row_sims, -(K + 1))[-(K + 1):]
            top_idx = top_idx[top_idx != global_idx]
            top_idx = top_idx[np.argsort(row_sims[top_idx])[::-1]][:K]
            top_k_neighbors[isbn_arr[global_idx]] = isbn_arr[top_idx].tolist()

    return books_clean, book_sim_df, top_k_neighbors, tfidf, content_matrix, content_books, ratings_clean


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


books_clean, book_sim_df, top_k_neighbors, tfidf, content_matrix, content_books, ratings_clean = load_data()
dl_model, mappings = load_dl_model()


# ── 中文书库（从 CSV 加载 + Open Library 搜索）──────────────────────────────

@st.cache_data
def load_cn_books():
    df = pd.read_csv('chinese_books.csv')
    df["ISBN"] = ["CN_" + str(i) for i in range(len(df))]
    df["Year-Of-Publication"] = ""
    df["Image-URL-M"] = ""
    df["avg_rating"] = None
    df["rating_count"] = 0
    df["source"] = "中文书库"
    return df

BUILTIN_CN_BOOKS = load_cn_books()

def _search_google_books(query, max_results=10):
    try:
        api_key = st.secrets["GOOGLE_BOOKS_API_KEY"]
    except Exception:
        return []
    url = f"https://www.googleapis.com/books/v1/volumes?q={query}&maxResults={max_results}&key={api_key}"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
    except Exception:
        return []
    rows = []
    for item in resp.json().get("items", []):
        info = item.get("volumeInfo", {})
        isbn = item.get("id", "GB_" + info.get("title", "")[:8])
        cover = (info.get("imageLinks") or {}).get("thumbnail", "")
        rows.append({
            "ISBN": isbn,
            "Book-Title": info.get("title", ""),
            "Book-Author": "、".join((info.get("authors") or ["未知作者"])[:2]),
            "Publisher": info.get("publisher", ""),
            "Year-Of-Publication": str(info.get("publishedDate", ""))[:4],
            "Image-URL-M": cover,
            "avg_rating": None,
            "rating_count": 0,
            "source": "Google Books",
            "description": info.get("description", "")[:200],
            "categories": "、".join((info.get("categories") or [])[:2]),
        })
    return rows


def search_chinese_books(query, max_results=20):
    mask = (
        BUILTIN_CN_BOOKS['Book-Title'].str.contains(query, case=False, na=False) |
        BUILTIN_CN_BOOKS['Book-Author'].str.contains(query, case=False, na=False) |
        BUILTIN_CN_BOOKS['categories'].str.contains(query, case=False, na=False)
    )
    builtin_results = BUILTIN_CN_BOOKS[mask].copy()

    # 若 CSV 结果不足 5 条，用 Google Books 补充
    gb_results = pd.DataFrame()
    if len(builtin_results) < 5:
        rows = _search_google_books(query, max_results=10)
        if rows:
            gb_results = pd.DataFrame(rows)

    combined = pd.concat([builtin_results, gb_results], ignore_index=True)
    combined = combined.drop_duplicates(subset='Book-Title')

    if combined.empty:
        return pd.DataFrame(), f"书库中未找到\"{query}\"，试试其他关键词"
    return combined.head(max_results), None


def get_google_books_recommendations(title, author, categories, num=10):
    """基于书的分类/作者找相似中文书。"""
    results = pd.DataFrame()

    if categories:
        cat = categories.split("、")[0]
        df, _ = search_chinese_books(cat, max_results=15)
        results = pd.concat([results, df], ignore_index=True)

    if author and author != "未知作者":
        df, _ = search_chinese_books(author, max_results=10)
        results = pd.concat([results, df], ignore_index=True)

    if not results.empty:
        results = results[results['Book-Title'] != title].drop_duplicates(subset='Book-Title')

    return results.head(num)


# ── 英文书内容邻居 ────────────────────────────────────────────────────────────

def get_content_neighbors(isbn, num):
    if isbn in top_k_neighbors:
        return top_k_neighbors[isbn][:num]
    row = books_clean[books_clean['ISBN'] == isbn]
    if row.empty:
        return []
    content = row.iloc[0]['Book-Author'] + ' ' + row.iloc[0]['Publisher']
    vec = tfidf.transform([content])
    sims = cosine_similarity(vec, content_matrix)[0]
    top_idx = np.argsort(sims)[::-1][:num]
    return content_books.iloc[top_idx]['ISBN'].tolist()


# ── 侧边栏 ────────────────────────────────────────────────────────────────────

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
    min_rating = st.slider("最低评分", 0.0, 10.0, 6.0, 0.5,
                           help="中文书评分数据较少，建议设为 0 以显示更多结果")


# ── 主搜索区 ──────────────────────────────────────────────────────────────────

col1, col2 = st.columns([3, 1])
with col1:
    keyword = st.text_input(
        "🔍 搜索书籍",
        placeholder="支持中英文：哈利波特 / Harry Potter / 明朝那些事",
    )
with col2:
    st.write("")
    st.write("")
    search_in = st.selectbox("搜索范围", ["书名", "作者", "全部"])


def is_chinese(text):
    return any('一' <= ch <= '鿿' for ch in text)


if keyword:
    keyword_stripped = keyword.strip()

    # ── 判断是中文还是英文搜索 ────────────────────────────────────────────────
    if is_chinese(keyword_stripped):
        # 中文：直接走 Google Books API
        with st.spinner("🔍 正在搜索中文书库..."):
            matched_cn, err = search_chinese_books(keyword_stripped, max_results=20)

        if matched_cn.empty:
            st.error(f"😢 没有找到相关书籍，试试其他关键词")
            if err:
                st.caption(f"调试信息：{err}")
        else:
            st.success(f"在中文书库找到 {len(matched_cn)} 本相关书籍")
            options = matched_cn['Book-Title'].tolist()
            selected = st.selectbox("📖 选择一本你喜欢的书：", options)

            if selected:
                book_row = matched_cn[matched_cn['Book-Title'] == selected].iloc[0]

                st.markdown("---")
                col1, col2 = st.columns([1, 3])
                with col1:
                    if book_row['Image-URL-M']:
                        st.image(book_row['Image-URL-M'], width=150)
                with col2:
                    st.markdown(f"### {book_row['Book-Title']}")
                    st.write(f"**作者:** {book_row['Book-Author']}")
                    if book_row['Publisher']:
                        st.write(f"**出版社:** {book_row['Publisher']}")
                    if book_row['avg_rating']:
                        st.write(f"**评分:** ⭐ {book_row['avg_rating']:.1f}/10 ({int(book_row['rating_count'])} 人评价)")
                    if book_row['description']:
                        st.write(f"**简介:** {book_row['description'][:200]}...")

                st.markdown("---")
                st.markdown(f"### 🎁 因为你喜欢《{selected}》，为你推荐：")

                with st.spinner("正在生成推荐..."):
                    recommendations = get_google_books_recommendations(
                        title=book_row['Book-Title'],
                        author=book_row['Book-Author'],
                        categories=book_row['categories'],
                        num=num_recs
                    )

                if recommendations.empty:
                    st.warning("😅 暂时没有找到相似书籍，试试更换搜索词")
                else:
                    for _, row in recommendations.iterrows():
                        col1, col2 = st.columns([1, 4])
                        with col1:
                            if row['Image-URL-M']:
                                st.image(row['Image-URL-M'], width=100)
                        with col2:
                            st.markdown(f"#### 📖 {row['Book-Title']}")
                            st.write(f"**作者:** {row['Book-Author']}")
                            if row['avg_rating']:
                                st.write(f"⭐ {row['avg_rating']:.1f}/10")
                            if row['description']:
                                st.write(f"{row['description'][:120]}...")
                        st.markdown("---")

    else:
        # 英文：走原有 Book-Crossing 数据集
        search_word = keyword_stripped.lower()

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
                        candidate_isbns = [b for b in list(book_sim_df.columns[:500])
                                           if b != isbn and b in mappings['book_to_idx']]

                        if valid_users and candidate_isbns:
                            u_idxs, b_idxs, b_isbns = [], [], []
                            for b_isbn in candidate_isbns:
                                b_idx = mappings['book_to_idx'][b_isbn]
                                for u in valid_users:
                                    u_idxs.append(mappings['user_to_idx'][u])
                                    b_idxs.append(b_idx)
                                    b_isbns.append(b_isbn)

                            with torch.no_grad():
                                preds = dl_model(
                                    torch.LongTensor(u_idxs),
                                    torch.LongTensor(b_idxs)
                                ).numpy()

                            scores = (
                                pd.DataFrame({'isbn': b_isbns, 'score': preds})
                                .groupby('isbn')['score'].mean()
                                .sort_values(ascending=False)
                            )
                            recommendations = books_clean[
                                books_clean['ISBN'].isin(scores.head(num_recs * 2).index)
                            ]
                        else:
                            st.warning("😅 没有足够的用户数据，已切换为混合推荐")
                            rec_method = "混合推荐（推荐）"

                if rec_method == "基于用户评分" and isbn in book_sim_df.columns:
                    similar = book_sim_df[isbn].sort_values(ascending=False)[1:num_recs + 1]
                    recommendations = books_clean[books_clean['ISBN'].isin(similar.index)]
                elif rec_method == "基于作者风格":
                    neighbor_isbns = get_content_neighbors(isbn, num_recs)
                    recommendations = books_clean[books_clean['ISBN'].isin(neighbor_isbns)]
                elif rec_method not in ("🤖 深度学习推荐",):
                    if isbn in book_sim_df.columns:
                        cf_similar = book_sim_df[isbn].sort_values(ascending=False)[1:num_recs * 2]
                        cf_books = books_clean[books_clean['ISBN'].isin(cf_similar.index)]
                    else:
                        cf_books = pd.DataFrame()

                    neighbor_isbns = get_content_neighbors(isbn, num_recs * 2)
                    cb_books = books_clean[books_clean['ISBN'].isin(neighbor_isbns)]
                    recommendations = pd.concat([cf_books, cb_books]).drop_duplicates(subset='ISBN').head(num_recs)

                if isinstance(recommendations, pd.DataFrame) and not recommendations.empty:
                    recommendations = recommendations[
                        (recommendations['avg_rating'] >= min_rating) |
                        (recommendations['avg_rating'].isna())
                    ]

                if isinstance(recommendations, pd.DataFrame) and recommendations.empty:
                    st.warning("😅 没有找到符合条件的推荐，试试降低最低评分要求")
                elif isinstance(recommendations, pd.DataFrame):
                    for _, row in recommendations.iterrows():
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
    st.info("💡 输入中文书名直接搜索（如：明朝那些事、大秦帝国），英文书名走原数据库")

    st.markdown("### 🔥 热门英文书籍")
    popular = books_clean.nlargest(6, 'rating_count', keep='first')
    cols = st.columns(3)
    for i, (_, row) in enumerate(popular.iterrows()):
        with cols[i % 3]:
            if pd.notna(row['Image-URL-M']):
                st.image(row['Image-URL-M'], width=150)
            title = row['Book-Title']
            st.write(f"**{title[:30]}...**" if len(title) > 30 else f"**{title}**")
            st.write(f"👤 {row['Book-Author']}")
            if pd.notna(row['avg_rating']):
                st.write(f"⭐ {row['avg_rating']:.1f}")

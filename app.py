import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

st.title("📚 书籍推荐系统")
st.write("输入你喜欢的书，找到相似的好书！")

@st.cache_data
def load_data():
    books = pd.read_csv('D:/book_project/Books.csv', low_memory=False)
    ratings = pd.read_csv('D:/book_project/Ratings.csv')
    books_clean = books.dropna(subset=['Book-Author', 'Publisher'])
    ratings_clean = ratings[ratings['Book-Rating'] > 0]
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
    return books_clean, book_sim_df

books_clean, book_sim_df = load_data()

book_input = st.text_input("输入书名（英文）", placeholder="例如：Sorcerer's Stone")

if book_input:
    matched = books_clean[books_clean['Book-Title'].str.contains(book_input, case=False, na=False)]
    if matched.empty:
        st.error("找不到这本书，请换个关键词试试")
    else:
        isbn = matched.iloc[0]['ISBN']
        title = matched.iloc[0]['Book-Title']
        if isbn not in book_sim_df.columns:
            st.warning(f"《{title}》数据不足，无法推荐")
        else:
            similar = book_sim_df[isbn].sort_values(ascending=False)[1:6]
            result = books_clean[books_clean['ISBN'].isin(similar.index)]
            st.success(f"因为你喜欢《{title}》，推荐你看：")
            for _, row in result.iterrows():
                st.write(f"📖 **{row['Book-Title']}** — {row['Book-Author']}")
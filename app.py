import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

st.title("📚 书籍推荐系统")
st.write("输入书名关键词，找到相似的好书！")

@st.cache_data
def load_data():
    books = pd.read_csv('Books.csv', low_memory=False)
    ratings = pd.read_csv('Ratings.csv')
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

# 搜索框
keyword = st.text_input("输入书名关键词（中英文均可）", placeholder="例如：魔法、Harry、Lord")

if keyword:
    # 翻译常见中文关键词
    zh_map = {
        "魔法": "magic", "哈利": "harry", "波特": "potter",
        "魔戒": "ring", "指环王": "ring", "小王子": "little prince",
        "夏洛": "charlotte", "纳尼亚": "narnia", "福尔摩斯": "holmes",
        "侦探": "detective", "爱情": "love", "战争": "war",
        "历史": "history", "科幻": "science", "奇幻": "fantasy"
    }
    search_word = keyword.lower()
    for zh, en in zh_map.items():
        if zh in keyword:
            search_word = en
            break

    # 模糊匹配书名
    matched = books_clean[books_clean['Book-Title'].str.contains(search_word, case=False, na=False)]

    if matched.empty:
        st.error("没有找到相关书籍，试试其他关键词")
    else:
        # 显示下拉选择框
        options = matched['Book-Title'].tolist()[:20]
        selected = st.selectbox("选择一本书：", options)

        if selected:
            isbn = matched[matched['Book-Title'] == selected].iloc[0]['ISBN']
            if isbn not in book_sim_df.columns:
                st.warning(f"《{selected}》数据不足，无法推荐")
            else:
                similar = book_sim_df[isbn].sort_values(ascending=False)[1:6]
                result = books_clean[books_clean['ISBN'].isin(similar.index)]
                st.success(f"因为你喜欢《{selected}》，推荐你看：")
                for _, row in result.iterrows():
                    st.write(f"📖 **{row['Book-Title']}** — {row['Book-Author']}")
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


# ── 中文书库（内置常见书 + Open Library 搜索）───────────────────────────────

# 内置常见中文书，保证基础搜索可用
BUILTIN_CN_BOOKS = pd.DataFrame([
    {"Book-Title": "明朝那些事儿（全集）", "Book-Author": "当年明月", "Publisher": "中国海关出版社", "categories": "历史", "description": "用现代语言讲述明朝三百年历史，风趣幽默，畅销千万册。"},
    {"Book-Title": "大秦帝国（全六部）", "Book-Author": "孙皓晖", "Publisher": "河南文艺出版社", "categories": "历史小说", "description": "宏大叙事还原战国时代秦国崛起的历史全景。"},
    {"Book-Title": "三体", "Book-Author": "刘慈欣", "Publisher": "重庆出版社", "categories": "科幻", "description": "中国科幻里程碑，讲述人类文明与三体文明的宏大碰撞。"},
    {"Book-Title": "三体II：黑暗森林", "Book-Author": "刘慈欣", "Publisher": "重庆出版社", "categories": "科幻", "description": "黑暗森林法则，宇宙文明生存的终极悖论。"},
    {"Book-Title": "三体III：死神永生", "Book-Author": "刘慈欣", "Publisher": "重庆出版社", "categories": "科幻", "description": "三体三部曲终章，人类文明的最终命运。"},
    {"Book-Title": "活着", "Book-Author": "余华", "Publisher": "作家出版社", "categories": "当代文学", "description": "一个人历经苦难依然活着的故事，震撼人心。"},
    {"Book-Title": "百年孤独", "Book-Author": "加西亚·马尔克斯", "Publisher": "南海出版公司", "categories": "外国文学", "description": "魔幻现实主义经典，布恩迪亚家族七代人的传奇。"},
    {"Book-Title": "红楼梦", "Book-Author": "曹雪芹", "Publisher": "人民文学出版社", "categories": "古典文学", "description": "中国四大名著之首，贾宝玉与林黛玉的爱情悲剧。"},
    {"Book-Title": "水浒传", "Book-Author": "施耐庵", "Publisher": "人民文学出版社", "categories": "古典文学", "description": "108位好汉聚义梁山，中国古典英雄传奇。"},
    {"Book-Title": "西游记", "Book-Author": "吴承恩", "Publisher": "人民文学出版社", "categories": "古典文学", "description": "孙悟空师徒四人西天取经的神话冒险。"},
    {"Book-Title": "三国演义", "Book-Author": "罗贯中", "Publisher": "人民文学出版社", "categories": "古典文学", "description": "魏蜀吴三国争霸，中国历史小说巅峰之作。"},
    {"Book-Title": "平凡的世界", "Book-Author": "路遥", "Publisher": "北京十月文艺出版社", "categories": "当代文学", "description": "陕北农村青年奋斗史，茅盾文学奖获奖作品。"},
    {"Book-Title": "围城", "Book-Author": "钱钟书", "Publisher": "人民文学出版社", "categories": "现代文学", "description": "婚姻如围城，深刻讽刺知识分子人性弱点。"},
    {"Book-Title": "白鹿原", "Book-Author": "陈忠实", "Publisher": "人民文学出版社", "categories": "当代文学", "description": "关中平原白鹿两家半个世纪的恩怨纠葛，茅盾文学奖。"},
    {"Book-Title": "追风筝的人", "Book-Author": "卡勒德·胡赛尼", "Publisher": "上海人民出版社", "categories": "外国文学", "description": "阿富汗背景下关于友谊、背叛与救赎的故事。"},
    {"Book-Title": "解忧杂货店", "Book-Author": "东野圭吾", "Publisher": "南海出版公司", "categories": "外国小说", "description": "时间交错中，杂货店的神秘信箱连接着迷途的心。"},
    {"Book-Title": "嫌疑人X的献身", "Book-Author": "东野圭吾", "Publisher": "南海出版公司", "categories": "推理", "description": "绝世天才为爱献身，东野圭吾最高杰作。"},
    {"Book-Title": "人类简史", "Book-Author": "尤瓦尔·赫拉利", "Publisher": "中信出版社", "categories": "历史科普", "description": "从认知革命到科学革命，重新审视人类进化全史。"},
    {"Book-Title": "未来简史", "Book-Author": "尤瓦尔·赫拉利", "Publisher": "中信出版社", "categories": "科普", "description": "算法与人工智能将如何重塑人类未来。"},
    {"Book-Title": "sapiens", "Book-Author": "尤瓦尔·赫拉利", "Publisher": "中信出版社", "categories": "历史科普", "description": "人类简史英文原版。"},
    {"Book-Title": "小王子", "Book-Author": "圣埃克苏佩里", "Publisher": "人民文学出版社", "categories": "童话", "description": "写给大人的童话，关于爱、责任与纯真。"},
    {"Book-Title": "挪威的森林", "Book-Author": "村上春树", "Publisher": "上海译文出版社", "categories": "外国小说", "description": "青春、爱情与死亡，村上春树代表作。"},
    {"Book-Title": "1984", "Book-Author": "乔治·奥威尔", "Publisher": "上海译文出版社", "categories": "反乌托邦", "description": "极权社会中的反抗与压迫，政治寓言经典。"},
    {"Book-Title": "动物森友会", "Book-Author": "任天堂", "Publisher": "任天堂", "categories": "游戏", "description": ""},
    {"Book-Title": "哈利·波特与魔法石", "Book-Author": "J.K.罗琳", "Publisher": "人民文学出版社", "categories": "奇幻", "description": "孤儿男孩哈利发现自己是巫师，进入霍格沃茨魔法学校。"},
    {"Book-Title": "哈利·波特与密室", "Book-Author": "J.K.罗琳", "Publisher": "人民文学出版社", "categories": "奇幻", "description": "霍格沃茨密室重开，学生接连被石化。"},
    {"Book-Title": "哈利·波特与阿兹卡班的囚徒", "Book-Author": "J.K.罗琳", "Publisher": "人民文学出版社", "categories": "奇幻", "description": "危险囚徒锡里斯·布莱克逃出阿兹卡班。"},
    {"Book-Title": "房思琪的初恋乐园", "Book-Author": "林奕含", "Publisher": "北京联合出版公司", "categories": "当代文学", "description": "以文学方式控诉性侵伤害，震撼台湾文坛。"},
    {"Book-Title": "芳华", "Book-Author": "严歌苓", "Publisher": "人民文学出版社", "categories": "当代文学", "description": "文工团一代人的青春、理想与命运。"},
    {"Book-Title": "盗墓笔记", "Book-Author": "南派三叔", "Publisher": "上海文化出版社", "categories": "悬疑冒险", "description": "吴邪跟随叔叔进入古墓探险的奇幻故事。"},
    {"Book-Title": "鬼吹灯", "Book-Author": "天下霸唱", "Publisher": "安徽文艺出版社", "categories": "悬疑冒险", "description": "胡八一等人盗墓探险系列，中国探险小说代表。"},
    {"Book-Title": "斗破苍穹", "Book-Author": "天蚕土豆", "Publisher": "朔方文化", "categories": "玄幻", "description": "废柴少年萧炎逆袭成神的修炼故事。"},
    {"Book-Title": "诛仙", "Book-Author": "萧鼎", "Publisher": "朔方文化", "categories": "仙侠", "description": "张小凡与碧瑶的爱情悲剧，仙侠经典。"},
    {"Book-Title": "凡人修仙传", "Book-Author": "忘语", "Publisher": "北方联合出版集团", "categories": "玄幻", "description": "普通农家子弟踏上修仙之路的漫长旅程。"},
    {"Book-Title": "庆余年", "Book-Author": "猫腻", "Publisher": "朔方文化", "categories": "历史玄幻", "description": "现代灵魂穿越古代王朝的权谋成长故事。"},
    {"Book-Title": "雪中悍刀行", "Book-Author": "烽火戏诸侯", "Publisher": "湖南文艺出版社", "categories": "武侠", "description": "徐凤年成长为天下第一的江湖传奇。"},
    {"Book-Title": "射雕英雄传", "Book-Author": "金庸", "Publisher": "广州出版社", "categories": "武侠", "description": "郭靖、黄蓉的成长与爱情，金庸武侠代表作。"},
    {"Book-Title": "天龙八部", "Book-Author": "金庸", "Publisher": "广州出版社", "categories": "武侠", "description": "萧峰、段誉、虚竹三位主角的命运交织。"},
    {"Book-Title": "笑傲江湖", "Book-Author": "金庸", "Publisher": "广州出版社", "categories": "武侠", "description": "令狐冲的逍遥人生与江湖权斗。"},
    {"Book-Title": "神雕侠侣", "Book-Author": "金庸", "Publisher": "广州出版社", "categories": "武侠", "description": "杨过与小龙女十六年分离重聚的爱情传奇。"},
    {"Book-Title": "苏东坡传", "Book-Author": "林语堂", "Publisher": "湖南文艺出版社", "categories": "传记", "description": "林语堂笔下的千古文人苏东坡。"},
    {"Book-Title": "曾国藩", "Book-Author": "唐浩明", "Publisher": "岳麓书社", "categories": "历史传记", "description": "晚清中兴名臣曾国藩的一生。"},
    {"Book-Title": "毛泽东传", "Book-Author": "罗斯·特里尔", "Publisher": "中国人民大学出版社", "categories": "传记", "description": "西方学者眼中的毛泽东生平与思想。"},
    {"Book-Title": "菜根谭", "Book-Author": "洪应明", "Publisher": "中华书局", "categories": "国学", "description": "修身处世的智慧格言集，流传数百年。"},
    {"Book-Title": "道德经", "Book-Author": "老子", "Publisher": "中华书局", "categories": "国学哲学", "description": "五千言道尽天地万物运行之道。"},
    {"Book-Title": "论语", "Book-Author": "孔子", "Publisher": "中华书局", "categories": "国学哲学", "description": "孔子及弟子言行录，儒家思想核心经典。"},
    {"Book-Title": "孙子兵法", "Book-Author": "孙武", "Publisher": "中华书局", "categories": "国学", "description": "世界最早的军事理论著作，影响深远。"},
    {"Book-Title": "乌合之众", "Book-Author": "古斯塔夫·勒庞", "Publisher": "中央编译出版社", "categories": "社会心理", "description": "群体心理学开山之作，解析大众行为规律。"},
    {"Book-Title": "影响力", "Book-Author": "罗伯特·西奥迪尼", "Publisher": "中国人民大学出版社", "categories": "心理学", "description": "六大影响力原则，揭示说服与顺从的秘密。"},
    {"Book-Title": "思考，快与慢", "Book-Author": "丹尼尔·卡尼曼", "Publisher": "中信出版社", "categories": "心理学", "description": "诺贝尔经济学奖得主解析人类思维的两套系统。"},
    {"Book-Title": "穷查理宝典", "Book-Author": "查理·芒格", "Publisher": "中信出版社", "categories": "投资", "description": "芒格思想智慧精华，投资与人生的多元思维模型。"},
    {"Book-Title": "聪明的投资者", "Book-Author": "本杰明·格雷厄姆", "Publisher": "人民邮电出版社", "categories": "投资", "description": "价值投资圣经，巴菲特最推崇的投资著作。"},
    {"Book-Title": "原则", "Book-Author": "瑞·达利欧", "Publisher": "中信出版社", "categories": "商业管理", "description": "桥水基金创始人的生活与工作原则。"},
    {"Book-Title": "乔布斯传", "Book-Author": "沃尔特·艾萨克森", "Publisher": "中信出版社", "categories": "传记", "description": "苹果创始人史蒂夫·乔布斯的唯一授权传记。"},
    {"Book-Title": "从0到1", "Book-Author": "彼得·蒂尔", "Publisher": "中信出版社", "categories": "商业", "description": "PayPal创始人讲述创业与创新的本质。"},
    {"Book-Title": "黑天鹅", "Book-Author": "纳西姆·塔勒布", "Publisher": "中信出版社", "categories": "思维", "description": "不可预测的极端事件如何改变世界。"},
    {"Book-Title": "数学之美", "Book-Author": "吴军", "Publisher": "人民邮电出版社", "categories": "科普", "description": "用数学解释自然语言处理与信息论之美。"},
    {"Book-Title": "浪潮之巅", "Book-Author": "吴军", "Publisher": "人民邮电出版社", "categories": "科技", "description": "硅谷科技公司兴衰史，IT行业深度解析。"},
    {"Book-Title": "失控", "Book-Author": "凯文·凯利", "Publisher": "电子工业出版社", "categories": "科技", "description": "机器、社会与经济的生物化，预言互联网时代。"},
    {"Book-Title": "深度工作", "Book-Author": "卡尔·纽波特", "Publisher": "江西人民出版社", "categories": "效率", "description": "专注是新时代最稀缺的能力。"},
    {"Book-Title": "刻意练习", "Book-Author": "安德斯·艾利克森", "Publisher": "机械工业出版社", "categories": "效率", "description": "天才不是天生的，正确的练习方法才是关键。"},
    {"Book-Title": "被讨厌的勇气", "Book-Author": "岸见一郎", "Publisher": "机械工业出版社", "categories": "心理学", "description": "阿德勒心理学，人生自由的勇气来源。"},
    {"Book-Title": "月亮与六便士", "Book-Author": "毛姆", "Publisher": "上海译文出版社", "categories": "外国文学", "description": "理想与现实的永恒抉择，画家高更的故事原型。"},
    {"Book-Title": "了不起的盖茨比", "Book-Author": "菲茨杰拉德", "Publisher": "上海译文出版社", "categories": "外国文学", "description": "美国梦的幻灭，爵士时代的浮华与悲剧。"},
    {"Book-Title": "麦田里的守望者", "Book-Author": "塞林格", "Publisher": "译林出版社", "categories": "外国文学", "description": "青少年叛逆经典，霍尔顿的彷徨与迷失。"},
    {"Book-Title": "局外人", "Book-Author": "加缪", "Publisher": "上海译文出版社", "categories": "外国文学", "description": "荒诞主义代表作，默尔索的疏离与冷漠。"},
    {"Book-Title": "变形记", "Book-Author": "卡夫卡", "Publisher": "上海译文出版社", "categories": "外国文学", "description": "人变甲虫的荒诞寓言，现代人异化的隐喻。"},
    {"Book-Title": "悲惨世界", "Book-Author": "雨果", "Publisher": "人民文学出版社", "categories": "外国文学", "description": "冉阿让的救赎之路，法国浪漫主义史诗。"},
    {"Book-Title": "战争与和平", "Book-Author": "列夫·托尔斯泰", "Publisher": "人民文学出版社", "categories": "外国文学", "description": "拿破仑战争时期俄国社会的全景画卷。"},
    {"Book-Title": "罪与罚", "Book-Author": "陀思妥耶夫斯基", "Publisher": "上海译文出版社", "categories": "外国文学", "description": "大学生拉斯科尔尼科夫的犯罪与救赎心理。"},
    {"Book-Title": "安娜·卡列尼娜", "Book-Author": "列夫·托尔斯泰", "Publisher": "上海译文出版社", "categories": "外国文学", "description": "爱情、婚姻与社会道德的永恒命题。"},
    {"Book-Title": "老人与海", "Book-Author": "海明威", "Publisher": "上海译文出版社", "categories": "外国文学", "description": "人可以被毁灭，但不可以被打败。"},
    {"Book-Title": "动物农场", "Book-Author": "乔治·奥威尔", "Publisher": "上海译文出版社", "categories": "外国文学", "description": "所有动物生而平等，但有些动物更平等。"},
    {"Book-Title": "美丽新世界", "Book-Author": "赫胥黎", "Publisher": "上海译文出版社", "categories": "反乌托邦", "description": "用娱乐与消费控制人类的未来反乌托邦。"},
    {"Book-Title": "福尔摩斯探案集", "Book-Author": "柯南·道尔", "Publisher": "人民文学出版社", "categories": "推理", "description": "夏洛克·福尔摩斯的经典侦探推理故事集。"},
    {"Book-Title": "无人生还", "Book-Author": "阿加莎·克里斯蒂", "Publisher": "新星出版社", "categories": "推理", "description": "孤岛上十个陌生人接连死去，侦探小说顶峰之作。"},
    {"Book-Title": "东方快车谋杀案", "Book-Author": "阿加莎·克里斯蒂", "Publisher": "新星出版社", "categories": "推理", "description": "密闭列车上的密室谋杀，波洛的绝世推理。"},
    {"Book-Title": "白夜行", "Book-Author": "东野圭吾", "Publisher": "南海出版公司", "categories": "推理", "description": "两个孩子命运交织，黑暗中的共生之爱。"},
    {"Book-Title": "恶意", "Book-Author": "东野圭吾", "Publisher": "南海出版公司", "categories": "推理", "description": "动机比凶手更难揭开，人性深处的恶意。"},
    {"Book-Title": "放学后", "Book-Author": "东野圭吾", "Publisher": "南海出版社", "categories": "推理", "description": "东野圭吾处女作，校园密室谋杀案。"},
    {"Book-Title": "十宗罪", "Book-Author": "蜘蛛", "Publisher": "重庆出版社", "categories": "悬疑推理", "description": "根据真实案例改编的犯罪悬疑故事。"},
    {"Book-Title": "法医秦明", "Book-Author": "秦明", "Publisher": "新星出版社", "categories": "悬疑推理", "description": "法医视角破解离奇命案，真实案例改编。"},
    {"Book-Title": "长安十二时辰", "Book-Author": "马伯庸", "Publisher": "湖南文艺出版社", "categories": "历史悬疑", "description": "大唐长安一天内反恐行动，节奏紧张。"},
    {"Book-Title": "古董局中局", "Book-Author": "马伯庸", "Publisher": "湖南文艺出版社", "categories": "悬疑", "description": "文物鉴定江湖的恩怨与谋局。"},
    {"Book-Title": "风声", "Book-Author": "麦家", "Publisher": "人民文学出版社", "categories": "谍战", "description": "抗战时期密码破译的极限智力博弈。"},
    {"Book-Title": "解密", "Book-Author": "麦家", "Publisher": "人民文学出版社", "categories": "谍战", "description": "天才密码学家容金珍的传奇一生。"},
    {"Book-Title": "繁花", "Book-Author": "金宇澄", "Publisher": "上海文艺出版社", "categories": "当代文学", "description": "沪语叙述上海几十年世俗生活，茅盾文学奖。"},
    {"Book-Title": "尘埃落定", "Book-Author": "阿来", "Publisher": "人民文学出版社", "categories": "当代文学", "description": "藏地土司制度的最后挽歌，茅盾文学奖。"},
    {"Book-Title": "额尔古纳河右岸", "Book-Author": "迟子建", "Publisher": "人民文学出版社", "categories": "当代文学", "description": "鄂温克族最后一个酋长女人的口述史。"},
    {"Book-Title": "许三观卖血记", "Book-Author": "余华", "Publisher": "作家出版社", "categories": "当代文学", "description": "用卖血换来的尊严与父爱。"},
    {"Book-Title": "兄弟", "Book-Author": "余华", "Publisher": "上海文艺出版社", "categories": "当代文学", "description": "两兄弟在时代变迁中的命运沉浮。"},
    {"Book-Title": "在细雨中呼喊", "Book-Author": "余华", "Publisher": "作家出版社", "categories": "当代文学", "description": "孤独少年在冷漠家庭中成长的记忆碎片。"},
    {"Book-Title": "蛙", "Book-Author": "莫言", "Publisher": "上海文艺出版社", "categories": "当代文学", "description": "计划生育政策下的生命悲歌，诺贝尔奖作品。"},
    {"Book-Title": "红高粱家族", "Book-Author": "莫言", "Publisher": "上海文艺出版社", "categories": "当代文学", "description": "高密东北乡的传奇抗日故事，莫言代表作。"},
    {"Book-Title": "丰乳肥臀", "Book-Author": "莫言", "Publisher": "作家出版社", "categories": "当代文学", "description": "中国现代史中一位母亲的苦难史诗。"},
    {"Book-Title": "边城", "Book-Author": "沈从文", "Publisher": "人民文学出版社", "categories": "现代文学", "description": "湘西小城翠翠的爱情悲剧，田园牧歌般的美丽。"},
    {"Book-Title": "呐喊", "Book-Author": "鲁迅", "Publisher": "人民文学出版社", "categories": "现代文学", "description": "阿Q正传、狂人日记等，中国现代文学开山之作。"},
    {"Book-Title": "朝花夕拾", "Book-Author": "鲁迅", "Publisher": "人民文学出版社", "categories": "现代文学", "description": "鲁迅回忆童年与青年的散文集。"},
    {"Book-Title": "傲慢与偏见", "Book-Author": "简·奥斯汀", "Publisher": "人民文学出版社", "categories": "外国文学", "description": "班纳特家五姐妹的婚姻故事，永恒的爱情喜剧。"},
    {"Book-Title": "简·爱", "Book-Author": "夏洛蒂·勃朗特", "Publisher": "人民文学出版社", "categories": "外国文学", "description": "孤女简·爱追求平等与尊严的爱情故事。"},
    {"Book-Title": "呼啸山庄", "Book-Author": "艾米莉·勃朗特", "Publisher": "人民文学出版社", "categories": "外国文学", "description": "希斯克利夫与凯瑟琳的爱恨纠缠，哥特式经典。"},
    {"Book-Title": "德伯家的苔丝", "Book-Author": "托马斯·哈代", "Publisher": "人民文学出版社", "categories": "外国文学", "description": "纯洁女子苔丝被命运摧毁的悲剧故事。"},
    {"Book-Title": "双城记", "Book-Author": "查尔斯·狄更斯", "Publisher": "人民文学出版社", "categories": "外国文学", "description": "法国大革命时期的爱情与牺牲。"},
    {"Book-Title": "基督山伯爵", "Book-Author": "大仲马", "Publisher": "人民文学出版社", "categories": "外国文学", "description": "爱德蒙·邓蒂斯的复仇与救赎传奇。"},
    {"Book-Title": "钢铁是怎样炼成的", "Book-Author": "奥斯特洛夫斯基", "Publisher": "人民文学出版社", "categories": "外国文学", "description": "保尔·柯察金的革命英雄主义成长故事。"},
    {"Book-Title": "静静的顿河", "Book-Author": "肖洛霍夫", "Publisher": "人民文学出版社", "categories": "外国文学", "description": "哥萨克人格里高利的战争与爱情史诗。"},
    {"Book-Title": "麦克白", "Book-Author": "莎士比亚", "Publisher": "人民文学出版社", "categories": "外国文学", "description": "权欲与野心导致的悲剧，莎翁四大悲剧之一。"},
    {"Book-Title": "哈姆雷特", "Book-Author": "莎士比亚", "Publisher": "人民文学出版社", "categories": "外国文学", "description": "生存还是毁灭，这是一个问题。"},
    {"Book-Title": "神曲", "Book-Author": "但丁", "Publisher": "人民文学出版社", "categories": "外国文学", "description": "穿越地狱、炼狱、天堂的中世纪史诗。"},
])

# 补全字段
BUILTIN_CN_BOOKS["ISBN"] = ["CN_" + str(i) for i in range(len(BUILTIN_CN_BOOKS))]
BUILTIN_CN_BOOKS["Year-Of-Publication"] = ""
BUILTIN_CN_BOOKS["Image-URL-M"] = ""
BUILTIN_CN_BOOKS["avg_rating"] = None
BUILTIN_CN_BOOKS["rating_count"] = 0
BUILTIN_CN_BOOKS["source"] = "中文书库"


def _search_openlibrary(query, max_results=15):
    url = "https://openlibrary.org/search.json"
    params = {"q": query, "limit": max_results,
              "fields": "key,title,author_name,cover_i,first_publish_year,publisher,subject"}
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    docs = resp.json().get("docs", [])
    rows = []
    for doc in docs:
        cover_id = doc.get("cover_i")
        cover_url = f"https://covers.openlibrary.org/b/id/{cover_id}-M.jpg" if cover_id else ""
        rows.append({
            "ISBN": doc.get("key", "").replace("/works/", "OL_"),
            "Book-Title": doc.get("title", ""),
            "Book-Author": "、".join(doc.get("author_name", ["未知作者"])[:2]),
            "Publisher": (doc.get("publisher") or [""])[0],
            "Year-Of-Publication": str(doc.get("first_publish_year", "")),
            "Image-URL-M": cover_url,
            "avg_rating": None,
            "rating_count": 0,
            "source": "中文书库",
            "description": "",
            "categories": "、".join((doc.get("subject") or [])[:3]),
        })
    return rows


def search_chinese_books(query, max_results=20):
    """先搜内置中文书库，再补充 Open Library 结果。"""
    q = query.strip().lower()

    # 内置库模糊匹配（书名 or 作者 or 分类）
    mask = (
        BUILTIN_CN_BOOKS['Book-Title'].str.contains(query, case=False, na=False) |
        BUILTIN_CN_BOOKS['Book-Author'].str.contains(query, case=False, na=False) |
        BUILTIN_CN_BOOKS['categories'].str.contains(query, case=False, na=False)
    )
    builtin_results = BUILTIN_CN_BOOKS[mask].copy()

    # Open Library 补充（出错也不影响内置结果）
    ol_results = pd.DataFrame()
    try:
        rows = _search_openlibrary(query, max_results=15)
        if rows:
            ol_results = pd.DataFrame(rows)
    except Exception:
        pass

    combined = pd.concat([builtin_results, ol_results], ignore_index=True)
    combined = combined.drop_duplicates(subset='Book-Title')

    if combined.empty:
        return pd.DataFrame(), f"内置书库和 Open Library 均未找到"{query}""
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

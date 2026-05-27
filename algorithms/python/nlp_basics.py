# -*- coding: utf-8 -*-
"""
NLP 基础工具 — Python 实现
功能：TF-IDF / LDA 主题模型 / Word2Vec / 文本预处理

用法：
    pip install scikit-learn gensim jieba
    python nlp_basics.py
"""

import numpy as np
import re


def preprocess_chinese(texts):
    """中文文本预处理：分词 + 去停用词"""
    try:
        import jieba
    except ImportError:
        print("[提示] jieba 未安装，使用简单空格分词")
        def jieba_cut(t): return t.split()

    # 常见停用词
    stopwords = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都',
                 '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你',
                 '会', '着', '没有', '看', '好', '自己', '这', '他', '她', '它',
                 '们', '那', '些', '所', '为', '所以', '因为', '但是', '虽然',
                 '可以', '这个', '如果', '还是', '就是', '又', '能', '而'}

    processed = []
    for text in texts:
        # 去除非中文字符
        text = re.sub(r'[^一-鿿]', ' ', str(text))
        # 分词
        words = [w.strip() for w in jieba.cut(text) if len(w.strip()) > 1]
        # 去停用词
        words = [w for w in words if w not in stopwords]
        processed.append(' '.join(words))

    return processed


def tfidf_analysis(documents, max_features=100):
    """
    TF-IDF 特征提取

    Returns:
        tfidf_matrix, feature_names, vectorizer
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(max_features=max_features, max_df=0.85, min_df=2)
    tfidf_matrix = vectorizer.fit_transform(documents)

    return {
        'matrix': tfidf_matrix,
        'feature_names': vectorizer.get_feature_names_out(),
        'vectorizer': vectorizer,
    }


def lda_topic_model(documents, n_topics=5, n_top_words=10):
    """
    LDA 主题模型

    Returns:
        topics (每个主题的前 n 个词), doc_topics (文档-主题分布), model
    """
    from sklearn.decomposition import LatentDirichletAllocation
    from sklearn.feature_extraction.text import CountVectorizer

    # 词频矩阵
    vectorizer = CountVectorizer(max_features=1000, max_df=0.85, min_df=2)
    dtm = vectorizer.fit_transform(documents)
    feature_names = vectorizer.get_feature_names_out()

    # LDA
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42,
                                    max_iter=20)
    doc_topics = lda.fit_transform(dtm)

    # 每个主题的词分布
    topics = {}
    for topic_idx, topic in enumerate(lda.components_):
        top_words_idx = topic.argsort()[-n_top_words:][::-1]
        top_words = [feature_names[i] for i in top_words_idx]
        topics[f'Topic {topic_idx+1}'] = top_words

    return {
        'topics': topics,
        'doc_topics': doc_topics,
        'model': lda,
        'perplexity': lda.perplexity(dtm),
    }


if __name__ == '__main__':
    print("=" * 60)
    print("NLP 基础工具示例")
    print("=" * 60)

    # 示例中文文档
    documents = [
        "数学建模竞赛中数据分析是非常重要的环节",
        "深度学习模型在图像识别领域取得了突破性进展",
        "层次分析法是一种常用的多准则决策方法",
        "卷积神经网络可以有效地提取图像特征",
        "TOPSIS法通过计算贴近度来对方案进行排序",
        "时间序列分析用于预测未来的数据趋势",
        "支持向量机是一种经典的机器学习分类算法",
        "模糊综合评价适合处理定性指标的评估问题",
        "LSTM网络擅长处理长序列的时序预测任务",
        "灰色预测模型适合小样本数据的短期预测",
    ]

    # 预处理
    print("\n[中文分词]")
    try:
        processed = preprocess_chinese(documents)
        for i, (orig, proc) in enumerate(zip(documents[:3], processed[:3])):
            print(f"  原文: {orig[:40]}...")
            print(f"  分词: {proc[:60]}...\n")
    except Exception as e:
        processed = documents
        print(f"  分词跳过: {e}")

    # TF-IDF
    tfidf_result = tfidf_analysis(processed if processed else documents)
    print(f"\n[TF-IDF] 特征词数: {len(tfidf_result['feature_names'])}")
    print(f"  前10个特征词: {list(tfidf_result['feature_names'][:10])}")

    # LDA 主题模型
    print(f"\n[LDA 主题模型]")
    lda_result = lda_topic_model(processed if processed else documents, n_topics=3, n_top_words=5)
    print(f"  Perplexity: {lda_result['perplexity']:.2f}")
    for topic_name, words in lda_result['topics'].items():
        print(f"  {topic_name}: {', '.join(words)}")

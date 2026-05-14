import os
import glob
import pickle

from langchain_classic.retrievers import ParentDocumentRetriever, EnsembleRetriever
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.stores import InMemoryStore
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableMap
from models import *

# 获得访问大模型和嵌入模型客户端
client, embeddings_model = get_ali_clients()

# 创建data目录用于持久化向量库
data_dir = "data"
os.makedirs(data_dir, exist_ok=True)

# 定义向量库路径
faiss_index_path = os.path.join(data_dir, "faiss_index")
# 父文档存储路径
docstore_path = os.path.join(data_dir, "docstore.pkl")

# 创建主文档分割器
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)

# 创建子文档分割器
child_splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=30)

# 检查向量库是否已存在
if os.path.exists(faiss_index_path) and os.path.exists(os.path.join(faiss_index_path, "index.faiss")):
    print("检测到已存在的向量库，正在加载...")

    # 从本地加载FAISS向量库
    vectorstore = FAISS.load_local(
        faiss_index_path,
        embeddings_model,
        allow_dangerous_deserialization=True
    )
    print(f"✓ 从本地加载FAISS向量库：{faiss_index_path}")
    print(f"✓ 向量库中的向量数量: {vectorstore.index.ntotal}")

    # 加载父文档存储
    docstore = InMemoryStore()
    if os.path.exists(docstore_path):
        with open(docstore_path, 'rb') as f:
            docstore_dict = pickle.load(f)
            docstore.mset(docstore_dict.items())
        print(f"✓ 从本地加载父文档存储：{docstore_path}")
        print(f"✓ 父文档数量: {len(docstore_dict)}")
    else:
        print("⚠ 警告：未找到父文档存储文件")

    # 创建ParentDocumentRetriever
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
        search_kwargs={"k": 3},
    )

    print("✓ 检索器已准备就绪...\n")

else:
    print("未检测到向量库，开始构建...\n")

    # 获取original目录下所有的docx文件
    docx_files = glob.glob("original/*.docx")
    print(f"找到 {len(docx_files)} 个文档文件")

    # 批量加载所有文档
    all_docs = []
    for file_path in docx_files:
        print(f"正在加载: {file_path}")
        loader = Docx2txtLoader(file_path)
        docs = loader.load()

        # 为每个文档添加元数据（文件名）
        file_name = os.path.basename(file_path)
        for doc in docs:
            doc.metadata['source_file'] = file_name

        all_docs.extend(docs)
        print(f"  - 加载完成")

    print(f"\n总共加载了 {len(all_docs)} 个文档对象")

    # 创建父文档存储
    docstore = InMemoryStore()

    # 创建空的FAISS向量库（用第一个文档的一小部分内容初始化）
    print("\n正在初始化向量库...")
    if all_docs:
        # 用第一个文档的前500字符创建初始向量
        init_doc = Document(page_content=all_docs[0].page_content[:500], metadata=all_docs[0].metadata)
        vectorstore = FAISS.from_documents(
            documents=[init_doc],
            embedding=embeddings_model
        )
        print(f"✓ 向量库初始化完成")

        # 创建ParentDocumentRetriever
        retriever = ParentDocumentRetriever(
            vectorstore=vectorstore,
            docstore=docstore,
            child_splitter=child_splitter,
            parent_splitter=parent_splitter,
            search_kwargs={"k": 3},
        )

        # 添加所有文档到retriever（让ParentDocumentRetriever自动处理父子分块）
        print("\n正在建立父子文档关系并添加文档...")
        retriever.add_documents(all_docs)
        print(f"✓ 文档已添加到ParentDocumentRetriever")
        print(f"✓ 向量库中的向量数量: {vectorstore.index.ntotal}")
    else:
        raise ValueError("没有找到任何文档可以处理")

    # 保存FAISS向量库
    print("\n正在保存向量库...")
    vectorstore.save_local(faiss_index_path)
    print(f"✓ FAISS向量库已保存到：{faiss_index_path}")
    print(f"✓ 向量库中的向量数量: {vectorstore.index.ntotal}")

    # 保存父文档存储
    with open(docstore_path, 'wb') as f:
        docstore_dict = dict(retriever.docstore.store)
        pickle.dump(docstore_dict, f)
    print(f"✓ 父文档存储已保存到：{docstore_path}")
    print(f"✓ 保存的父文档数量: {len(docstore_dict)}\n")

# ==================== 混合检索功能 ====================
print("=" * 50)
print("初始化混合检索系统")
print("=" * 50)

# 获取所有文档用于BM25
print("\n正在初始化BM25关键词检索器...")
all_docx_files = glob.glob("original/*.docx")
bm25_docs = []
for file_path in all_docx_files:
    loader = Docx2txtLoader(file_path)
    docs = loader.load()
    file_name = os.path.basename(file_path)
    for doc in docs:
        doc.metadata['source_file'] = file_name
    bm25_docs.extend(docs)

# 用子分割器分割文档用于BM25
bm25_splits = child_splitter.split_documents(bm25_docs)
bm25_retriever = BM25Retriever.from_documents(bm25_splits)
bm25_retriever.k = 3
print("BM25关键词检索器初始化完成")

# 向量检索器
vector_retriever = retriever

# 创建混合检索器
hybrid_retriever = EnsembleRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    weights=[0.6, 0.4],  # 向量检索权重60%，关键词检索权重40%，生产环境，可能关键词权重可能还要再低些
)
print("混合检索器已就绪（向量 + 关键词）")

# 可选：初始化重排序模型（如果需要使用）
try:
    reranker = get_ali_rerank(top_n=3)
    use_rerank = True
    print("重排序模型已就绪\n")
except Exception as e:
    use_rerank = False
    print(f"重排序模型不可用: {e}\n")

# ==================== 查询功能 ====================
print("=" * 50)
print("开始查询测试")
print("=" * 50)


def multi_stage_retrieval(query, hybrid_retriever, reranker=None, use_rerank=False):
    """
    多阶段检索流程：
    1. 混合检索（向量 + 关键词）
    2. 去重合并
    3. 重排序（可选）
    """
    # 第一阶段：混合检索
    retrieved_docs = hybrid_retriever.invoke(query)

    # 第二阶段：去重
    seen_contents = set()
    unique_docs = []
    for doc in retrieved_docs:
        content_key = doc.page_content[:100]
        if content_key not in seen_contents:
            seen_contents.add(content_key)
            unique_docs.append(doc)

    # 第三阶段：重排序（可选）
    if use_rerank and reranker and len(unique_docs) > 1:
        try:
            reranked_docs = reranker.compress_documents(unique_docs, query)
            return reranked_docs
        except Exception as e:
            print(f"重排序失败: {e}，使用原始排序")

    return unique_docs


def check_relevance(query, docs, threshold=0.5):
    """
    检查检索结果与查询的相关性

    Args:
        query: 用户查询
        docs: 检索到的文档列表
        threshold: 相关性阈值（0-1），越高要求越严格

    Returns:
        bool: 是否相关
        float: 最高相似度分数
    """
    if not docs:
        return False, 0.0

    # 获取第一个（最相关的）文档
    top_doc = docs[0]

    # 如果有score属性，直接使用
    if hasattr(top_doc, 'metadata') and 'score' in top_doc.metadata:
        score = top_doc.metadata['score']
        return score >= threshold, score

    # 否则通过嵌入模型计算相似度
    try:
        # 计算查询和文档的嵌入向量
        query_embedding = embeddings_model.embed_query(query)
        doc_embedding = embeddings_model.embed_documents([top_doc.page_content])[0]

        # 计算余弦相似度
        import numpy as np
        query_vec = np.array(query_embedding)
        doc_vec = np.array(doc_embedding)

        similarity = np.dot(query_vec, doc_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(doc_vec))

        return similarity >= threshold, float(similarity)
    except Exception as e:
        print(f"相似度计算失败: {e}")
        # 默认认为相关，避免误判
        return True, 1.0


def generate_answer_with_relevance_check(query, relevant_docs, threshold=0.5):
    """
    带相关性检查的答案生成

    Args:
        query: 用户查询
        relevant_docs: 检索到的相关文档
        threshold: 相关性阈值

    Returns:
        str: 回答内容
    """
    # 如果没有检索到文档
    if not relevant_docs:
        return "抱歉，我在知识库中没有找到相关信息来回答您的问题。"

    # 检查相关性
    is_relevant, similarity_score = check_relevance(query, relevant_docs, threshold)

    print(f"\n📊 相关性分析:")
    print(f"   - 最高相似度: {similarity_score:.4f}")
    print(f"   - 阈值设定: {threshold}")
    print(f"   - 判断结果: {'✅ 相关' if is_relevant else '❌ 不相关'}")

    if not is_relevant:
        return f"抱歉，您的问题与知识库中的内容关联度较低（相似度: {similarity_score:.2%}），我可能无法给出准确的回答。建议您：\n1. 换一种提问方式\n2. 询问与法律相关的问题\n3. 提供更具体的关键词"

    # 创建prompt模板（添加相关性指令）
    template = """请根据下面给出的上下文来回答问题。如果上下文与问题无关，请明确说明。

上下文:
{context}

问题: {question}

请基于上述上下文提供准确的回答。如果上下文不足以回答问题，请诚实地告知用户。
回答:"""

    # 由模板生成prompt
    prompt = ChatPromptTemplate.from_template(template)

    # 创建chain
    chain = RunnableMap({
        "context": lambda x: "\n\n".join([doc.page_content for doc in relevant_docs[:2]]),  # 使用前2个最相关文档
        "question": lambda x: x["question"]
    }) | prompt | client | StrOutputParser()

    return chain.invoke({"question": query})


# 交互式查询
while True:
    user_query = input("\n请输入您的问题: ").strip()

    if user_query.lower() in ['quit', 'exit', 'q']:
        print("感谢使用，再见！")
        break

    if not user_query:
        continue

    print(f"\n正在搜索: {user_query}")

    # 使用混合检索
    relevant_docs = multi_stage_retrieval(
        user_query,
        hybrid_retriever,
        reranker if use_rerank else None,
        use_rerank
    )

    print(f"\n找到 {len(relevant_docs)} 个相关结果（混合检索" + (" + 重排序" if use_rerank else "") + "):\n")
    for i, doc in enumerate(relevant_docs, 1):
        print(f"[{i}] 来源: {doc.metadata.get('source_file', '未知')}")
        print(f"    内容: {doc.page_content[:300]}...")

    print("=" * 50)
    print("=" * 50)

    # 带相关性检查的答案生成
    print("\n------------向量检索+BM25 -> 相关性检查 -> 大模型回答------------------------")
    answer = generate_answer_with_relevance_check(user_query, relevant_docs, threshold=0.5)
    print(answer)
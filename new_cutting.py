import os
import glob
import pickle

from langchain_classic.retrievers import ParentDocumentRetriever, EnsembleRetriever
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.stores import InMemoryStore
from langchain_community.vectorstores import FAISS
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


def incremental_update(new_folder="new"):
    """
    增量更新向量库和docstore
    :param new_folder: 存放新文档的文件夹路径
    :return: 是否更新成功
    """
    print("\n" + "=" * 50)
    print("✓ 开始增量更新向量库")
    print("=" * 50)

    # 检查新文档文件夹是否存在
    if not os.path.exists(new_folder):
        print(f"警告：{new_folder} 文件夹不存在，跳过增量更新")
        return False

    # 获取新文档文件
    new_docx_files = glob.glob(os.path.join(new_folder, "*.docx"))
    if not new_docx_files:
        print(f"警告：{new_folder} 文件夹中没有找到 .docx 文件")
        return False

    print(f"✓ 找到 {len(new_docx_files)} 个新文档文件")

    # 加载新文档
    new_docs = []
    for file_path in new_docx_files:
        print(f"正在加载: {file_path}")
        loader = Docx2txtLoader(file_path)
        docs = loader.load()

        # 添加元数据
        file_name = os.path.basename(file_path)
        for doc in docs:
            doc.metadata['source_file'] = file_name

        new_docs.extend(docs)
        print(f"  - 加载完成")

    print(f"\n✓ 总共加载了 {len(new_docs)} 个新文档对象")

    # 加载现有的向量库
    print("\n✓ 正在加载现有向量库...")
    vectorstore = FAISS.load_local(
        faiss_index_path,
        embeddings_model,
        allow_dangerous_deserialization=True
    )
    print(f"✓ 现有向量数量: {vectorstore.index.ntotal}")

    # 加载现有的 docstore
    docstore = InMemoryStore()
    old_docstore_count = 0
    if os.path.exists(docstore_path):
        with open(docstore_path, 'rb') as f:
            docstore_dict = pickle.load(f)
            old_docstore_count = len(docstore_dict)
            docstore.mset(docstore_dict.items())
        print(f"✓ 已加载现有父文档存储，数量: {old_docstore_count}")

    # 创建 ParentDocumentRetriever
    temp_retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )

    # 添加新文档到 retriever（会自动进行父子分块并添加到向量库）
    print("\n正在处理新文档的分割和索引...")
    temp_retriever.add_documents(new_docs)

    new_vector_count = vectorstore.index.ntotal - vectorstore.index.ntotal
    print(f"新文档处理完成")
    print(
        f"更新后向量总数: {vectorstore.index.ntotal}（新增 {vectorstore.index.ntotal - vectorstore.index.ntotal} 个）")

    new_docstore_count = len(temp_retriever.docstore.store)
    print(f"更新后父文档总数: {new_docstore_count}（新增 {new_docstore_count - old_docstore_count} 个）")

    # 保存更新后的向量库
    print("\n正在保存更新后的向量库...")
    vectorstore.save_local(faiss_index_path)
    print(f"FAISS向量库已保存到: {faiss_index_path}")

    # 保存更新后的 docstore
    with open(docstore_path, 'wb') as f:
        docstore_dict = dict(temp_retriever.docstore.store)
        pickle.dump(docstore_dict, f)
    print(f"父文档存储已保存到: {docstore_path}")
    print(f"保存的父文档总数: {len(docstore_dict)}")

    # 验证：测试查询一个新文档中的关键词
    print("\n" + "=" * 50)
    print("验证：测试查询新文档内容")
    print("=" * 50)

    if new_docs:
        # 取新文档中的一小段内容进行测试
        test_content = new_docs[0].page_content[:50]
        print(f"\n使用新文档片段进行测试查询: \"{test_content}...\"")

        test_results = temp_retriever.invoke(test_content[:30])
        print(f"查询结果数量: {len(test_results)}")

        if test_results:
            print("✓ 验证成功！新文档可以被查询到")
            print(f"  第一条结果来源: {test_results[0].metadata.get('source_file', '未知')}")
        else:
            print("⚠ 警告：查询测试失败，新文档可能未被正确索引")

    print("\n" + "=" * 50)
    print("✓ 增量更新完成！")
    print("=" * 50)

    return True


if __name__ == "__main__":
    incremental_update()
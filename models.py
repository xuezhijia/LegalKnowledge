# models.py
#可用模型列表，以及获得访问模型的客户端
#实际使用时可以根据自己的实际情况调整
ALI_TONGYI_API_KEY_OS_VAR_NAME = "DASHSCOPE_API_KEY"
ALI_TONGYI_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
ALI_TONGYI_MAX_MODEL = "qwen-max"
ALI_TONGYI_DEEPSEEK_R1 = "deepseek-r1"
ALI_TONGYI_DEEPSEEK_V3 = "deepseek-v3"
ALI_TONGYI_REASONER_MODEL = "qwq-plus"
ALI_TONGYI_EMBEDDING_MODEL = "text-embedding-v3"
ALI_TONGYI_RERANK_MODEL = "gte-rerank-v2"

DEEPSEEK_API_KEY_OS_VAR_NAME = "Deepseek_Key"
DEEPSEEK_URL = "https://api.deepseek.com/v1"
DEEPSEEK_CHAT_MODEL = "deepseek-chat"
DEEPSEEK_REASONER_MODEL = "deepseek-reasoner"

import os
from langchain_openai import ChatOpenAI
import inspect
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.document_compressors.dashscope_rerank import DashScopeRerank

def get_lc_model_client(api_key=os.getenv(ALI_TONGYI_API_KEY_OS_VAR_NAME), base_url=ALI_TONGYI_URL
                        , model=ALI_TONGYI_MAX_MODEL, temperature=0.7,verbose=False, debug=False):
    '''
    过LangChain获得指定平台和模型的客户端
    可以通过传入api_key，base_url，model，temperature四个参数来覆盖默认值
    verbose，debug两个参数，分别控制是否输出调试信息，是否输出详细调试信息，默认不打印
    :return: 指定平台和模型的客户端，默认平台和模型为腾讯混元，温度=0.7
    '''
    function_name = inspect.currentframe().f_code.co_name
    if(verbose):
        print(f"{function_name}-平台：{base_url},模型：{model},温度：{temperature}")
    if(debug):
        print(f"{function_name}-平台：{base_url},模型：{model},温度：{temperature},key：{api_key}")
    return ChatOpenAI(api_key=api_key, base_url=base_url,model=model,temperature=temperature)

def get_ali_model_client(model=ALI_TONGYI_MAX_MODEL,temperature = 0.7,verbose=False, debug=False):
    '''
    过LangChain获得阿里大模型的客户端
    可以通过传入model，temperature 两个参数来覆盖默认值
    verbose，debug两个参数，分别控制是否输出调试信息，是否输出详细调试信息，默认不打印
    :return: 指定平台和模型的客户端，默认模型为阿里百炼里的qwen-max-latest，温度=0.7
    '''
    return get_lc_model_client(api_key=os.getenv(ALI_TONGYI_API_KEY_OS_VAR_NAME), base_url=ALI_TONGYI_URL
                        ,model=model,temperature =temperature,verbose=verbose, debug=debug )

def get_ds_model_client(model=DEEPSEEK_CHAT_MODEL,temperature = 0.7,verbose=False, debug=False):
    '''
    过LangChain获得DeepSeek大模型的客户端
    可以通过传入model，temperature 两个参数来覆盖默认值
    verbose，debug两个参数，分别控制是否输出调试信息，是否输出详细调试信息，默认不打印
    :return: 指定平台和模型的客户端，默认模型为DeepSeek的deepseek-chat，温度=0.7
    '''
    return get_lc_model_client(api_key=os.getenv(DEEPSEEK_API_KEY_OS_VAR_NAME), base_url=DEEPSEEK_URL
                        ,model=model,temperature =temperature,verbose=verbose, debug=debug )

def get_ali_embeddings():
    '''
    通过LangChain获得一个阿里通义千问嵌入模型的实例
    :return: 阿里通义千问嵌入模型的实例，目前为text-embedding-v3
    '''
    return DashScopeEmbeddings(
        model=ALI_TONGYI_EMBEDDING_MODEL,
        dashscope_api_key=os.getenv(ALI_TONGYI_API_KEY_OS_VAR_NAME)
    )

def get_ali_clients():
    '''
    产生阿里大模型客户端和嵌入模型的客户端
    :return: 阿里大模型客户端和嵌入模型的客户端
    '''
    return get_ali_model_client(),get_ali_embeddings()


def get_ali_rerank(top_n=3):
    '''
    通过LangChain获得一个阿里重排序模型的实例
    :return: 阿里通义千问嵌入模型的实例
    '''
    return DashScopeRerank(
        model=ALI_TONGYI_RERANK_MODEL, dashscope_api_key=os.getenv(ALI_TONGYI_API_KEY_OS_VAR_NAME),
        top_n=top_n
    )
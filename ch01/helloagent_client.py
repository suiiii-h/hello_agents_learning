import os#跟操作系统打交道，读环境变量
from openai import OpenAI
from dotenv import load_dotenv#读环境变量，把里面的键值对塞进环境变量
from typing import List,Dict#给变量标类型说明
#加载.env中的环境变量
load_dotenv()

class HelloAgentsLLM:
    """
    为hello agent 定制的LLM客户端，用于任何兼容openai接口的服务，使用流式输出
    """
    def __init__(self,model =None,api_key = None,base_url = None,timeout = None):#为什么要给默认值？or要先判断左边为false再去判断右边，
                                                                                #也就是model等三个参数都为默认值再去读环境变量，所以没有默认值就永远不会执行右边的语句
        self.model = model or os.environ.get("LLM_MODEL_ID")
        api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        base_url = base_url or os.environ.get("LLM_BASE_URL")

        if not all([self.model,api_key,base_url]):
            raise ValueError("模型，api密钥，服务地址必须同时提供，.env中定义！")#把报错向上上报
        self.client = OpenAI(api_key=api_key,base_url=base_url,timeout=timeout)

    def think(self,messages:List[Dict[str,str]],temperature :float = 0)->str:
        """调用大模型并且接收返回结果，采用流式输出
        """
        print(f"大模型正在运行中，调用{self.model}")
        try:
            response = self.client.chat.completions.create(
                model = self.model,
                messages = messages,
                temperature = temperature,
                stream = True,
            )
            #处理流式输出
            print("大模型响应成功！")
            full_response = []
            for chunk in response:
                if not chunk.choices:
                    continue
                content = chunk.choices[0].delta.content or ""
                print(content,end = "",flush = True)#flush强制print立即输出，不会在缓冲区攒输出
                full_response.append(content)
            print()#换行
            return "".join(full_response)#用空字符串链接这个list里面的字符串

        except Exception as e:
            print(f"调用llm出错了，错误原因：{e}")

#客户端使用案例兼测试
if __name__ == '__main__':
    try:
        LLMClient = HelloAgentsLLM()
        examplemessage = [{"role":"system","content":"你是可爱小鲸鱼"},
                          {"role":"user","content":"输出一段100字得自我介绍"}]
        print("调用LLM")
        response = LLMClient.think(examplemessage)
        if response:
            print("\n\n---完整模型响应---")
            print(response)
    except ValueError as e:
        print(e)



            
                
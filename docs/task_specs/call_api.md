# API调用需求文档

请帮我实现一个用于数据蒸馏的 Python 框架，要求是“简单、易读、鲁棒、方便后续自己改”，不要过度工程化。

目标：(作为最基本的功能，是不是应该放在utils文件夹里？最优应该如何设计？)
- 一个非常易用、简单且鲁棒的API调用代码：
- 调用 OpenAI-compatible chat completions API
- 支持 retry /  timeout / 错误保留
- 集成已有的 log_completion(model, request_data, response_data, user)，注意这个代码现在没有提供（未来会把源码复制粘贴过来），调用方法见后面的例子

- 针对具体任务来进行API调用，比如：
  - 应用1：最简单demo：直接手动指定文本query & 图片路径，直接调用一次API得到模型返回结果。这样我可以快速在一个简单代码文件里快速debug API
  - 应用2：根据我们已经筛选好种子query，提取出来，并行调用API进行蒸馏，支持 resume /并行

总的来讲，用户主要只需要配置：
1. API / model 配置：base_url, api_key, model_name, timeout, retries
2. 输入内容构造器：纯文本、图文、多图，且图文顺序可控
然后就可以调用模型得到返回结果了，后续可以自己再对结果做提取&处理

设计原则：
- 优先简单和白盒，不要做复杂抽象
- 本期只支持 OpenAI-compatible API（包括本地 vLLM serve 和第三方兼容接口）

对于应用2，我们是要专门设计一个蒸馏的文件夹用于SVG蒸馏：
必须支持：
- 读取已有筛选出来的jsonl（参考之前处理文件：/home/yanghaitao/Projects/Data/query_seed/SAgoge/high_priority_pool.jsonl ），把`instruction` 这个field的文本提取出来，设计最优的user prompt用于svg蒸馏。
- resume：记录调用过程，中途失败后还可以重新resume
- retry：支持 timeout / 429 / 5xx / connection error 重试
- 流式写出：每条完成后 append 到 jsonl，避免中途挂掉丢结果
- 并行
- 错误记录：失败样本也必须写出 status 和 error 信息
- 日志：log_completion 失败不能影响主流程

一个代码例子：
函数接口&内容都可以修改，这里主要是想告诉你已有代码 log_completion (一个内部代码)如何调用，它可以用来全局统一记录调用记录信息
```
import os
import requests
import json
import httpx
from local_api_logger import log_completion


def call_api(url, api_key, model_name, user_content):
    # get message content from user_content
    # user_content = [
    #     {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
    #     {"type": "text", "text": user_prompt}
    # ]
     
    try:
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": user_content,
                }
            ],
            "model": model_name,
        }

        headers = {
            "Content-Type": "application/json",
            'Authorization': f"Bearer {api_key}",
        }

        # 调用API（超时时间120秒）
        response_obj = requests.post(
            url, 
            headers=headers, 
            json=payload, 
            verify=False, 
            timeout=120
        )
        print(response_obj.status_code)
        print(response_obj.text)
        response_obj.raise_for_status()  # 捕获HTTP错误（4xx/5xx）

        # 解析响应
        response = response_obj.json()
        log_completion(
            model=model_name,
            request_data=payload,
            response_data=response,
            user='visual_coding'
        )

        # 提取并清理GPT返回内容
        content = response["choices"][0]["message"]["content"]

    except Exception as e:
        print(str(e))
```
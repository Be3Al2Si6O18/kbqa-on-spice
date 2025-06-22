import time
import asyncio
from openai import OpenAI
from tqdm import tqdm, trange
from openai import AsyncOpenAI
from aiolimiter import AsyncLimiter
from tqdm.asyncio import tqdm_asyncio
# from langchain_openai import ChatOpenAI
from typing import List, Tuple, TypedDict, Union, Optional

API_KEY = 'sk-54964e5c3b8c4998a74f7d3e35b618ac'
API_KEYS = ['sk-54964e5c3b8c4998a74f7d3e35b618ac', 'sk-785362add9834d1da5907f4621dc90a8'] # list of keys
API_URL = 'https://api.deepseek.com'
API_MODEL = 'deepseek-chat'

@staticmethod
async def run_task_with_progress(task, pbar):
    result = await task
    pbar.update(1)
    return result

@staticmethod
def failure_mode(response):
    # TBD
    return False
        
class AsyncLLM:
    def __init__(
            self,
            api_model: str = "deepseek-chat",
            api_url: str = "https://api.deepseek.com",
            num_per_second: int = 100,
            api_key: str = API_KEY,
            use_async = True,
            openai_params: Optional[dict] = {},
            **kwargs,
        ):
        self.api_model: str = api_model
        self.api_url: str = api_url
        self.api_key: str = api_key
        self.num_per_second: int = num_per_second
        self.limiter = AsyncLimiter(self.num_per_second, 1)
        self.retry_times: int = 3
        self.openai_params: Optional[dict] = openai_params
        if use_async:
            self.llm = AsyncOpenAI(
                base_url=self.api_url, api_key=self.api_key, **kwargs
            )
        else:
            self.llm = OpenAI(
                base_url=self.api_url, api_key=self.api_key, **kwargs
            )
        # for counting
        self.succ = 0
        self.fail = 0

    async def _async_invoke(self, messages, **kwargs):
        
        for _ in range(self.retry_times):
            try:
                response = await self.llm.chat.completions.create(
                    model=self.api_model,
                    messages=messages,
                    stream=False,
                    **self.openai_params
                )

                # TODO
                if failure_mode(response):
                    self.fail += 1
                    continue
                # Success
                else:
                    self.succ += 1
                    return response
            except Exception:
                continue
            return ''
        
    async def __call__(self, messages):
        # 限速
        async with self.limiter:
            return await self._async_invoke(messages)

class LLM_Call:

    def __init__(
        self,
        api_key: Union[bool, str] = API_KEY,
        api_url: Union[bool, str] = API_URL,
        api_model: Union[bool, str] = "deepseek-chat",
        api_pool: Union[bool, List[Tuple[str, str, str]]] = False,
        use_async_api: bool = True,
        num_per_second: int = 10,
        openai_params: Optional[dict] = {},   # used for create API calls
        **kwargs   # used for initializing AsyncLLM
    ):
        '''
        API Params:
            api_pool: a list of (api_key, api_url, api_model)
            use_async_api: 
            num_per_second:

        OpenAI Params:
            temperature: float = 1.0,
            presence_penality: float = 0.0,
            frequency_penality: float = 0.0,
            max_tokens: int = 2048,
        '''
        self.model = api_model
        self.use_async_api = use_async_api
        self.num_per_second = num_per_second
        self.limiter = AsyncLimiter(self.num_per_second, 1)
        self.clients = []
        self.openai_params: Optional[dict] = openai_params

        if api_pool == False:
            self.clients.append(
                AsyncLLM(api_model=api_model, 
                         api_key=api_key, 
                         api_url=api_url, 
                         num_per_second=num_per_second, 
                         use_async=use_async_api, 
                         openai_params=openai_params, 
                         **kwargs)
            )
        else:
            print(api_pool)
            for k, u, m in api_pool:
                self.clients.append(
                    AsyncLLM(api_key=k, 
                             api_url=u, 
                             api_model=m, 
                             num_per_second=num_per_second, 
                             use_async=use_async_api, 
                             openai_params=openai_params, 
                             **kwargs)
                )

    # No async
    def single_chat(self, client_index=0, content="Hello"):
        response = self.clients[client_index].llm.chat.completions.create(
            model=self.model,
            messages=[
                # {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": content},
            ],
            stream=False
        )
        result = response.choices[0].message.content
        return result

    # async batch inference
    async def _batch_generate_async(self, messages_list: List[List]):
        results = [self.clients[index % len(self.clients)](messages) for index, messages in enumerate(messages_list)]
        with tqdm(total=len(results)) as pbar:
            results = await asyncio.gather(
                *[run_task_with_progress(task, pbar) for task in results]
            )
        return results

    # no async batch inference
    def _batch_generate(self, prompts: List[str]) -> Tuple[List[str]]:
        results = []
        s = time.time()
        for index, prompt in enumerate(tqdm(prompts)):
            response = self.single_chat(index % len(self.clients), prompt)
            results.append(response)
        print(time.time() - s)
        return results


if __name__ == "__main__":
    api_pool = [(k, API_URL, API_MODEL) for k in API_KEYS]
    LLM = LLM_Call(api_pool=api_pool, openai_params = {'temperature': 0, 'max_tokens': 512})
    prompts = [
        "解释量子力学",
        "写一首关于秋天的诗",
        "如何学习Python？"
    ] * 3

    messages_list = [[{'role': 'user', 'content': prompt}] for prompt in prompts]

    # jupyter
    # res = await LLM._batch_generate_async(messages_list)
    # print(res)

    # terminal
    res = asyncio.run(LLM._batch_generate_async(messages_list))
    print(len(res))

import yfinance as yf
import litellm

from aegis.templates.template import TemplateRegistry
from aegis.templates.complexity_dept.complexity_agent import (
    complexity_agent,
    complexity_dependencies,
)
from aegis.pre_agent import pre_agent, post_agent


# uv run pytest tests/test_litellm.py::test_llm -v
def test_llm() -> None:
    response = litellm.completion(
        model="deepseek/deepseek-chat",
        messages=[{"role": "user", "content": "hello from litellm"}],
        api_key="sk-7399448825f04cbeb8bce541f8cbcdcb",
        stream=True,
    )
    for chunk in response:
        print(chunk)


def test_yf() -> None:
    ticker = yf.Ticker("AAPL")
    info = ticker.info
    import pdb

    pdb.set_trace()


# uv run pytest tests/test_litellm.py::test_template -v
def test_template() -> None:
    TemplateRegistry.get("complexity_dept")


# 此处需要传入具体的ts_code，需要有专门具体的agent去读取意图，再分发给这个组件
# 后处理需要将结论输出一个结果描述，发送给用户
# uv run pytest tests/test_litellm.py::test_complexity_agent -v
def test_complexity_agent() -> None:
    user_promot = "请分析一下宏辉果蔬（股票代码：603336）"
    pre_result = pre_agent.run_sync(user_promot)

    result = complexity_agent.run_sync(
        user_promot,
        deps=pre_result.output,
    )
    print(result.output)
    post_result = post_agent.run_sync(
        user_promot,
        deps=result.output,
    )
    print(post_result.output)
    import pdb

    pdb.set_trace()

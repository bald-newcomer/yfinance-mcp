from typing import get_args

from yfinance.const import SECTOR_INDUSTY_MAPPINGs
from yfmcp.types import Sector

import yfinance as yf
import litellm
from aegis.templates.template import TemplateRegistry

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
    ticker = yf.Ticker('AAPL')
    info = ticker.info
    import pdb; pdb.set_trace()

def test_template() -> None:
    TemplateRegistry.get()

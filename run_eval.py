import json
import logging

from logging_config import configure_logging

configure_logging()

from langchain_core.messages import HumanMessage
from main_graph import app

logger = logging.getLogger(__name__)


def run_evaluation(data_path):
    # 1. 加载题库
    with open(data_path, 'r', encoding='utf-8') as f:
        test_cases = json.load(f)
    
    results = []
    
    # 2. 循环测试
    for case in test_cases:
        question = case["question"]
        expected = case["key_point"]
        
        # 调用你的 Agent
        result = app.invoke({"messages": [HumanMessage(content=question)]})
        response = result["messages"][-1].content
        
        # 3. 裁判逻辑 (Judge)
        is_passed = expected in response
        results.append({"question": question, "passed": is_passed, "response": response})

        logger.info(
            "题目: %s\n结果: %s",
            question,
            "通过" if is_passed else "失败",
        )
        logger.info(
            "题目: %s\n预期: %s\nAI实际回答: %s",
            question,
            expected,
            response,
        )
    pass_rate = sum(1 for r in results if r["passed"]) / len(results)
    logger.info("总测试通过率: %.1f%%", pass_rate * 100)


if __name__ == "__main__":
    run_evaluation("eval_data.json")
    
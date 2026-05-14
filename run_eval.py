import json
from main_graph import app # 从你的入口文件导入 app
from langchain_core.messages import HumanMessage

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
        
        print(f"题目: {question}\n结果: {'✅ 通过' if is_passed else '❌ 失败'}\n")
        print(f"题目: {question}\n预期: {expected}\nAI实际回答: {response}\n结果: ...")
    # 4. 输出简易报告
    pass_rate = sum(1 for r in results if r["passed"]) / len(results)
    print(f"总测试通过率: {pass_rate:.1%}")
    # 在你的 run_eval.py 的 for 循环里加一行
    
if __name__ == "__main__":
    # 请确保根目录下有 eval_data.json
    run_evaluation("eval_data.json")
    
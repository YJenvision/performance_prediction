import json
from intent_recognition_routing.intent_recognizer import SteelPerformanceIntentRecognizer
from steel_automl.performance_model_builder import performanceModelBuilder

if __name__ == "__main__":

    recognizer_instance = SteelPerformanceIntentRecognizer()
    example_requests_for_direct_test = [
        "用过去两年Q235B牌号的数据构建一个抗拉强度预报模型，使用XGboost，用贝叶斯优化利用R2进行超参数优化，尽可能使得模型效果好，可以创建一些合理的多项式或者比率交互特征等来增强模型效果，我的可接受误差浮动是正负20以内，不需要进行异常值的识别和处理。",
        # "用过去两年H033和H043机组，AR4162E7和AR3162E1出钢记号的相关数据构建一个抗拉强度性能预报模型，使用XGboost，对于ELM_B、ELM_PB_ACT特征列的缺失值使用众数进行填充，使用随机搜索进行超参数优化，快速验证一下，不进行异常值的判别和处理。",
        # "用过去一年的数据构建一个H033机组Q235B的抗拉强度的性能预报模型，使用随机森林算法，我觉得可以使用断裂延伸率辅助预测，使用随机搜索进行超参数优化，尽可能使得模型效果好，不进行异常值的判别和处理。"
    ]

    for req in example_requests_for_direct_test:
        output = recognizer_instance.recognize_intent(req)
        print(f"意图识别与路由阶段结果：\n{json.dumps(output, indent=2, ensure_ascii=False, default=str)}")

        results = performanceModelBuilder(output)
        print("\n建模与评估工程结果：\n")
        print(json.dumps(results, indent=2, ensure_ascii=False, default=str))

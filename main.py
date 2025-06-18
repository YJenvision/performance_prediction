import json
from intent_recognition_routing.intent_recognizer import SteelPerformanceIntentRecognizer
from steel_automl.performance_model_builder import performanceModelBuilder

if __name__ == "__main__":

    recognizer_instance = SteelPerformanceIntentRecognizer()
    example_requests_for_direct_test = [
        # "用过去一年的数据构建一个H033机组Q235B的屈服延伸率的性能预报模型，使用随机森林算法，不进行异常值的判别和处理。",
        # "用过去一年的数据构建一个H033机组抗拉强度的性能预报模型，使用XGboost，使用贝叶斯优化进行超参数优化，尽可能使得模型效果好，不进行异常值的判别和处理。",
        "用过去一年的数据构建一个H033机组抗拉强度的性能预报模型，使用XGboost，使用贝叶斯优化进行超参数优化，尽可能使得模型效果好，不进行异常值的判别和处理。",
        # "用过去一年的数据构建一个H033和H043机组，AR4162E7和AR3162E1出钢记号的抗拉强度性能预报模型，使用XGboost，使用随机搜索进行超参数优化，快速验证一下，不进行异常值的判别和处理。",
    ]
    for req in example_requests_for_direct_test:
        output = recognizer_instance.recognize_intent(req)
        # output = {
        #     "user_request": "我需要为H033机组预测抗拉强度，数据时间范围是2024年1月1日到2025年1月1日，不需要进行异常数据的判别和处理。",
        #     "sg_sign": "Q235B",
        #     "target_metric": "TS_N",
        #     "time_range": "20250101-20250501",
        #     "product_unit_no": None,
        #     "st_no": None
        # }
        print(f"开始AutoML流程，请求参数\n{json.dumps(output, indent=4, ensure_ascii=False, default=str)}")
        results = performanceModelBuilder(output)

        print("\n\n===== AutoML流程最终返回结果 =====")
        print(json.dumps(results, indent=4, ensure_ascii=False, default=str))  # default=str 用于处理无法序列化的对象如datetime

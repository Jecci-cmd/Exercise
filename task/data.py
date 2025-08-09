# 文件名: generate_data.py
import pandas as pd
import random

def generate_problem_by_max_digits(max_d1, max_d2):
    """
    根据给定的最大位数，随机生成一个加法问题。
    这会产生不对齐的例子，例如 max_d1=3, max_d2=3,
    可能会生成 1位数+3位数, 2位数+2位数, 3位数+1位数 等组合。
    """
    d1 = random.randint(1, max_d1)
    d2 = random.randint(1, max_d2)
    
    # 为了让数据更丰富，有70%的概率取到最大位数
    if random.random() < 0.7:
        d1 = max_d1
    if random.random() < 0.7:
        d2 = max_d2
        
    num1 = random.randint(0, 10**d1 - 1)
    num2 = random.randint(0, 10**d2 - 1)
    
    result = num1 + num2
    return f"{num1}+{num2}={result}"

def generate_final_dataset(total_samples, output_filename):
    """
    生成一个内容多样、不重复的数据集。
    """
    print(f"开始生成最终版数据集，总目标样本数: {total_samples}...")
    
    all_problems = set() # 使用全局集合来保证所有问题的唯一性
    
    # 定义不同难度等级的最大位数组合
    # 这有助于确保我们能生成足够多样性的数据
    level_definitions = [
        (1, 1), (2, 2), (3, 3), (4, 4), (5, 5),
        (1, 2), (1, 3), (1, 4), (1, 5),
        (2, 3), (2, 4), (2, 5),
        (3, 4), (3, 5),
        (4, 5)
    ]

    while len(all_problems) < total_samples:
        max_d1, max_d2 = random.choice(level_definitions)
        
        problem = generate_problem_by_max_digits(max_d1, max_d2)
        all_problems.add(problem)
        
        # 打印进度
        if len(all_problems) % 500 == 0 and len(all_problems) > 0:
            print(f"  已生成 {len(all_problems)} / {total_samples} 条不重复数据...")

    print(f"\n总共生成 {len(all_problems)} 条数据。")
    
    final_problems_list = list(all_problems)
    random.shuffle(final_problems_list) # 在保存前打乱一下顺序

    df = pd.DataFrame(final_problems_list, columns=['text'])
    
    df.to_csv(output_filename, index=False)
    
    print(f"最终数据集已成功保存至 {output_filename}")

if __name__ == "__main__":
    TOTAL_SAMPLES = 5000
    OUTPUT_FILENAME = 'addition_dataset_final.csv'
    
    generate_final_dataset(
        total_samples=TOTAL_SAMPLES,
        output_filename=OUTPUT_FILENAME
    )
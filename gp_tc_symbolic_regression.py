
import sys
sys.setrecursionlimit(10000)  # 提高递归深度，防止编译树时出现 MemoryError

import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 指定中文字体
matplotlib.rcParams['axes.unicode_minus'] = False    # 正常显示负号

import random, math, operator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from deap import base, creator, gp, tools
from functools import partial
import os
import time

# -------------------------
# 定义安全运算函数
def safeDiv(a, b):
    try:
        return a / b if abs(b) > 1e-6 else 1.0
    except Exception:
        return 1.0

def safeLog(x):
    try:
        return math.log(abs(x)) if abs(x) > 1e-6 else 0.0
    except Exception:
        return 0.0

# -------------------------
# 构建 GP 原始函数集，4 个输入变量：mass, pressure, hydrogen, mu_star
pset = gp.PrimitiveSet("MAIN", 4)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(safeDiv, 2)
pset.addPrimitive(math.sin, 1)
pset.addPrimitive(math.cos, 1)
pset.addPrimitive(safeLog, 1)
pset.addEphemeralConstant("rand101", partial(random.uniform, -1, 1))
pset.renameArguments(ARG0='mass')
pset.renameArguments(ARG1='pressure')
pset.renameArguments(ARG2='hydrogen')
pset.renameArguments(ARG3='mu_star')

# -------------------------
# 避免重复创建类
if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

# -------------------------
# 定义 GP 个体的适应度评估函数
# 采用10折交叉验证，每折内部以训练集90%/测试集10%计算GP模型输出的x与Tc的皮尔逊相关系数（取绝对值）
def eval_individual(individual, X, y, cv_splits=10):
    # 尝试编译表达式，如遇 MemoryError 返回极低适应度
    try:
        func = toolbox.compile(expr=individual)
    except MemoryError:
        return -9999,
    
    kf = KFold(n_splits=cv_splits, shuffle=True, random_state=42)
    r_values = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        try:
            x_test = np.array([func(*row) for row in X_test])
        except Exception:
            return -9999,
        if np.std(x_test) == 0 or np.std(y_test) == 0:
            r = 0
        else:
            r = np.corrcoef(x_test, y_test)[0, 1]
        r_values.append(abs(r))
    return np.mean(r_values),

toolbox.register("evaluate", eval_individual, X=None, y=None)  # 稍后固定 X, y 参数
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

# -------------------------
# 添加bloat控制：限制树的最大高度，防止树过深导致内存错误
MAX_HEIGHT = 17
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=MAX_HEIGHT))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=MAX_HEIGHT))

# -------------------------
# GP进化过程：单次循环，种群规模 pop_size，进化 ngen 代（例如：5000 代）
def gp_evolution_single(X, y, ngen=5000, pop_size=20, print_freq=100):
    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)
    start_time = time.time()
    for gen in range(ngen):
        # 计算未评估个体的适应度
        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        for ind in invalid_ind:
            ind.fitness.values = toolbox.evaluate(ind, X=X, y=y)
        best_fit = max(ind.fitness.values[0] for ind in pop)
        if gen % print_freq == 0:
            print(f"Generation {gen} best CV相关系数 (绝对值): {best_fit:.4f}", flush=True)
        # 保留精英2
        elite = tools.selBest(pop, 2)
        offspring = list(elite)
        # 交配操作8
        for _ in range(8):
            ind1, ind2 = toolbox.select(pop, 2)
            child = toolbox.clone(ind1)
            toolbox.mate(child, ind2)
            del child.fitness.values
            offspring.append(child)
        # 变异操作8
        for _ in range(8):
            ind = toolbox.clone(random.choice(pop))
            toolbox.mutate(ind)
            del ind.fitness.values
            offspring.append(ind)
        # 随机生成2个新个体
        for _ in range(2):
            offspring.append(toolbox.individual())
        # 保证种群大小为 pop_size
        offspring = offspring[:pop_size]
        pop = offspring
        hof.update(pop)
    elapsed = time.time() - start_time
    print(f"单次GP进化耗时: {elapsed:.2f} 秒")
    best_ind = hof[0]
    return best_ind, best_ind.fitness.values[0]
def final_evaluation_and_plot(best_model, X, y):
    # 修改 test_size 为 0.1，使训练集占 90%，测试集占 10%
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    
    func = toolbox.compile(expr=best_model)
    x_train = np.array([func(*row) for row in X_train]).reshape(-1, 1)
    x_test  = np.array([func(*row) for row in X_test]).reshape(-1, 1)
    
    lr = LinearRegression().fit(x_train, y_train)
    a = lr.coef_[0]
    b = lr.intercept_
    
    y_pred = lr.predict(x_test)
    if np.std(y_pred)==0 or np.std(y_test)==0:
        R = 0
    else:
        R = np.corrcoef(y_pred, y_test)[0,1]
    
    gp_expr_str = str(best_model)
    print("\n最佳 GP 表达式（x的特征函数）：")
    print(gp_expr_str)
    print("\n基于测试集数据拟合的线性回归函数：")
    print(f"Tc = {a:.3f} * x + {b:.3f}")
    print(f"测试集最大相关系数: {R:.3f}")
    
    output_dir = r"C:\Users\81246\Desktop\文献\机器学习综述👆\Mg超导体\数据集文献\5000代输出文件"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_txt = os.path.join(output_dir, "final_results.txt")
    with open(output_txt, "w", encoding="utf-8") as f:
        f.write("最佳 GP 表达式（x的特征函数）：\n")
        f.write(gp_expr_str + "\n\n")
        f.write("基于测试集数据拟合的线性回归函数：\n")
        f.write(f"Tc = {a:.3f} * x + {b:.3f}\n")
        f.write(f"测试集最大相关系数: {R:.3f}\n")
    
    plt.figure(figsize=(10,6))
    # 这里可以调整散点大小，使得视觉效果统一
    plt.scatter(x_train, y_train, color='blue', marker='o', s=50, alpha=0.7, label="训练集")
    plt.scatter(x_test, y_test, color='green', marker='s', s=50, alpha=0.7, label="测试集")
    
    x_all = np.concatenate((x_train, x_test)).reshape(-1,1)
    x_grid = np.linspace(x_all.min(), x_all.max(), 100).reshape(-1,1)
    y_grid = lr.predict(x_grid)
    plt.plot(x_grid, y_grid, color='red', linewidth=2, label="拟合直线")
    
    plt.xlabel("x = F(mass, pressure, hydrogen, mu_star)")
    plt.ylabel("Tc (K)")
    plt.title(f"测试集上 x 与 Tc 的线性拟合\n测试集相关系数: {R:.3f}")
    plt.legend()
    plt.grid(True)
    
    eq_text = f"Tc = {a:.2f} * x + {b:.2f}"
    plt.text(0.05, 0.95, f"GP表达式：\n{gp_expr_str}\n\n线性回归：\n{eq_text}",
             transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plot_path = os.path.join(output_dir, "final_plot4.png")
    plt.savefig(plot_path)
    print(f"\n图像已保存到: {plot_path}")
    plt.show()
    plt.close()

    
    # 在图中添加文本框显示 GP 表达式和线性回归公式
    eq_text = f"Tc = {a:.2f} * x + {b:.2f}"
    plt.text(0.05, 0.95, f"GP表达式：\n{gp_expr_str}\n\n线性回归：\n{eq_text}",
             transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 保存图像到指定目录
    plot_path = os.path.join(output_dir, "final_plot1.png")
    plt.savefig(plot_path)
    print(f"\n图像已保存到: {plot_path}")
    plt.show()
    plt.close()

# -------------------------
# 主流程
if __name__ == '__main__':
    # 读取Excel数据（本论文数据集）
    file_path = r"C:\Users\81246\Desktop\文献\机器学习综述👆\Mg超导体\数据集文献\ca-mg-h数据集.xlsx"
    df = pd.read_excel(file_path)
    df.columns = df.columns.str.strip()
    df.rename(columns={
        'mass': 'mass',
        'pressure(Gpa)': 'pressure',
        'H_concentration': 'hydrogen',
        'μ': 'mu_star',
        'Tc(K)': 'Tc'
    }, inplace=True)
    df = df[['mass', 'pressure', 'hydrogen', 'mu_star', 'Tc']]
    df.dropna(inplace=True)
    
    # 对于数值列，如有百分号则剥离并转换为浮点数
    num_cols = ['mass', 'pressure', 'hydrogen', 'mu_star']
    for col in num_cols:
        if df[col].dtype == object:
            df[col] = df[col].str.rstrip('%')
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['Tc'] = pd.to_numeric(df['Tc'], errors='coerce')
    df.dropna(inplace=True)
    
    # 特征标准化
    scaler = StandardScaler()
    X = scaler.fit_transform(df[num_cols].values)
    y = df['Tc'].values
    
    # 重新注册 evaluate 固定 X 和 y 参数
    toolbox.unregister("evaluate")
    toolbox.register("evaluate", eval_individual, X=X, y=y)
    
    # 这里采用顺序运行多次 GP 进化，避免并行可能引起的问题
    runs = 50 
    best_model = None
    best_cv_score = -9999
    for i in range(runs):
        print(f"\n===== 第 {i+1} 次GP进化 =====")
        model, score = gp_evolution_single(X, y, ngen=5000, pop_size=20, print_freq=100)
        print(f"第 {i+1} 次运行的交叉验证相关系数 (绝对值): {score:.4f}")
        if score > best_cv_score:
            best_cv_score = score
            best_model = model
    
    print("\n多次运行后，最佳模型为：")
    print(f"交叉验证相关系数 (绝对值): {best_cv_score:.4f}")
    print("最佳 GP 表达式（x的特征函数）：")
    print(best_model)
    
    # 最终评估、输出并保存结果和图像
    final_evaluation_and_plot(best_model, X, y)

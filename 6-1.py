# %%
# 训练决策树并其可视化
# 决策树只需要很少样本就可以生成，而且不需要对特征进行缩放
# 而且 Scikit 使用的为 CART algorithm，即每次只生成两个分支
# 而ID3等算法可以产生多个分支
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

iris = load_iris()
X = iris.data[:, 2:]  # petal length and width
y = iris.target
tree_clf = DecisionTreeClassifier(max_depth=2)
tree_clf.fit(X, y)
pp = tree_clf.predict_proba([[5, 1.5]])
p = tree_clf.predict([[5, 1.5]])
print(pp, p)

from sklearn.tree import export_graphviz
export_graphviz(
    tree_clf,
    out_file="iris_tree.dot",
    feature_names=iris.feature_names[2:],
    class_names=iris.target_names,
    rounded=True,
    filled=True
)

# %%
# 正则化参数（Regularization Hyperparameters）
# 由于决策树算法对训练数据没有什么假设（相比线性模型假设决策线为一条线）
# 这就对算法没有任何限制，因此很容易拟合训练数据，从而很容易导致过拟合
# 因此为了防止对训练数据过拟合，需要增加一些参数来限制
# 最一般的设置应该设置最大深度（max_depth）
# min_samples_split min_samples_leaf min_weight_fraction_leaf
# max_leaf_nodes max_features
# 增加 min_*，减小 max_* 都能正则化算法

# %%
# 决策树回归（Decision Tree Regression）
import numpy as np
from sklearn.tree import DecisionTreeRegressor

# Quadratic training set + noise
np.random.seed(42)
m = 200
X = np.random.rand(m, 1)
y = 4 * (X - 0.5) ** 2

tree_reg = DecisionTreeRegressor(max_depth=2, random_state=42)
tree_reg.fit(X, y)

# %%
# 局限性（不稳定性）
# 虽然决策树易于使用，易于展示与理解，但是存在一些局限性
# 决策树喜欢正交决策边界（垂直），这使得它们对训练集旋转很敏感
# 如果训练样本稍有变动，可能会导致决策线发生巨大改变
# 决策树算法非常不稳定
# 随机森林可以通过对多个决策树作平均来限制这种不稳定性

# %%
# 练习题 7
from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

X, y = make_moons(n_samples=10000, noise=0.4, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
params = {'max_leaf_nodes': list(
    range(2, 100)), 'min_samples_split': [2, 3, 4]}
grid_search_cv = GridSearchCV(DecisionTreeClassifier(
    random_state=42), params, n_jobs=-1, verbose=1, cv=5)
grid_search_cv.fit(X_train, y_train)
y_pred = grid_search_cv.predict(X_test)
score = accuracy_score(y_test, y_pred)
print(score)

# %%
# 练习题 8
from scipy.stats import mode
from sklearn.base import clone
from sklearn.model_selection import ShuffleSplit

n_trees = 1000
n_instances = 100
mini_sets = []
rs = ShuffleSplit(n_splits=n_trees, test_size=len(
    X_train) - n_instances, random_state=42)
for mini_train_index, mini_test_index in rs.split(X_train):
    X_mini_train = X_train[mini_train_index]
    y_mini_train = y_train[mini_train_index]
    mini_sets.append((X_mini_train, y_mini_train))

forest = [clone(grid_search_cv.best_estimator_) for _ in range(n_trees)]

accuracy_scores = []
for tree, (X_mini_train, y_mini_train) in zip(forest, mini_sets):
    tree.fit(X_mini_train, y_mini_train)
    y_pred = tree.predict(X_test)
    accuracy_scores.append(accuracy_score(y_test, y_pred))

score = np.mean(accuracy_scores)
print(score)

Y_pred = np.empty([n_trees, len(X_test)], dtype=np.uint8)
for tree_index, tree in enumerate(forest):
    Y_pred[tree_index] = tree.predict(X_test)

y_pred_majority_votes, n_votes = mode(Y_pred, axis=0)
score = accuracy_score(y_test, y_pred_majority_votes.reshape([-1]))
print(score)

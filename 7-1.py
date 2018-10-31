# %%
# 假设要解决一个复杂的问题，让众多学生去回答，然后汇总他们的答案
# 在许多情况下，会发现这个汇总的答案比一个老师的答案要好
# 如果汇总了一组预测变量（例如分类器或回归因子）的预测结果
# 则通常会得到比最佳个体预测变量得到更好的预测结果
# 这种技术被称为集成学习（Ensemble Learning）
# 投票分类器（Voting Classifiers ）
# 硬投票(hard voting)
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

X, y = make_moons(n_samples=500, noise=0.30, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

log_clf = LogisticRegression(random_state=42, solver='lbfgs')
rnd_clf = RandomForestClassifier(random_state=42, n_estimators=100)
svm_clf = SVC(random_state=42, gamma='auto')

voting_clf = VotingClassifier(
    estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)], voting='hard')

voting_clf.fit(X_train, y_train)

for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))

# %%
# 软投票(soft voting)
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

X, y = make_moons(n_samples=500, noise=0.30, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

log_clf = LogisticRegression(random_state=42, solver='lbfgs')
rnd_clf = RandomForestClassifier(random_state=42, n_estimators=100)
svm_clf = SVC(probability=True, random_state=42, gamma='auto')

voting_clf = VotingClassifier(
    estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)], voting='soft')

voting_clf.fit(X_train, y_train)

for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))

# %%
# Bagging and Pasting
# 对每个分类器使用相同的算法，但是要在训练集的不同随机子集上进行训练
# 如果抽样时有放回，称为Bagging；当抽样没有放回，称为Pasting
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

bag_clf = BaggingClassifier(
    DecisionTreeClassifier(random_state=42), n_estimators=500,
    max_samples=100, bootstrap=True, n_jobs=-1, random_state=42)
bag_clf.fit(X_train, y_train)
y_pred = bag_clf.predict(X_test)
print(accuracy_score(y_test, y_pred))

# Out-of-Bag Evaluation
bag_clf = BaggingClassifier(
    DecisionTreeClassifier(), n_estimators=500,
    bootstrap=True, n_jobs=-1, oob_score=True)
bag_clf.fit(X_train, y_train)
print("oob", bag_clf.oob_score_)
y_pred = bag_clf.predict(X_test)
print("test", accuracy_score(y_test, y_pred))

# %%
# 随机森林（Random forest）
from sklearn.ensemble import RandomForestClassifier
rnd_clf = RandomForestClassifier(
    n_estimators=500, max_leaf_nodes=16, n_jobs=-1)
rnd_clf.fit(X_train, y_train)
y_pred_rf = rnd_clf.predict(X_test)
print("test", accuracy_score(y_test, y_pred_rf))

# %%
# 特征重要性(Feature Importance)
from sklearn.datasets import load_iris

iris = load_iris()
rnd_clf = RandomForestClassifier(n_estimators=500, n_jobs=-1)
rnd_clf.fit(iris["data"], iris["target"])
for name, score in zip(iris["feature_names"], rnd_clf.feature_importances_):
    print(name, score)

# %%
# Boosting
# Boosting 是将弱学习器集成为强学习器的方法，主要思想是按顺序训练学习器，以尝试修改之前的学习器
# Boosting 的方法有许多，最为有名的方法为 AdaBoost（Adaptive Boosting）和 Gradient Boosting
# AdaBoost
# 一个新的学习器会更关注之前学习器分类错误的训练样本
# 因此新的学习器会越来越多地关注困难的例子
from sklearn.ensemble import AdaBoostClassifier

ada_clf = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=1), n_estimators=200,
    algorithm="SAMME.R", learning_rate=0.5, random_state=42)
ada_clf.fit(X_train, y_train)

# Gradient Boosting
# 和 AdaBoost 类似，Gradient Boosting 也是逐个训练学习器，尝试纠正前面学习器的错误
# 不同的是，AdaBoost 纠正错误的方法是更加关注前面学习器分错的样本
# Gradient Boosting（适合回归任务）纠正错误的方法是拟合前面学习器的残差（预测值减真实值）
import numpy as np
from sklearn.tree import DecisionTreeRegressor

np.random.seed(42)
X = np.random.rand(100, 1) - 0.5
y = 3*X[:, 0]**2 + 0.05 * np.random.randn(100)

tree_reg1 = DecisionTreeRegressor(max_depth=2, random_state=42)
tree_reg1.fit(X, y)

y2 = y - tree_reg1.predict(X)
tree_reg2 = DecisionTreeRegressor(max_depth=2, random_state=42)
tree_reg2.fit(X, y2)

y3 = y2 - tree_reg2.predict(X)
tree_reg3 = DecisionTreeRegressor(max_depth=2, random_state=42)
tree_reg3.fit(X, y3)

X_new = np.array([[0.8]])
y_pred = sum(tree.predict(X_new) for tree in (tree_reg1, tree_reg2, tree_reg3))
print(y_pred)

from sklearn.ensemble import GradientBoostingRegressor
gbrt = GradientBoostingRegressor(
    max_depth=2, n_estimators=3, learning_rate=1.0)
gbrt.fit(X, y)
y_pred = gbrt.predict(X_new)
print(y_pred)

# %%
# Gradient Boosting with Early stopping
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=49)

gbrt = GradientBoostingRegressor(
    max_depth=2, n_estimators=120, random_state=42)
gbrt.fit(X_train, y_train)

errors = [mean_squared_error(y_val, y_pred)
          for y_pred in gbrt.staged_predict(X_val)]
bst_n_estimators = np.argmin(errors)

gbrt_best = GradientBoostingRegressor(
    max_depth=2, n_estimators=bst_n_estimators, random_state=42)
gbrt_best.fit(X_train, y_train)

gbrt = GradientBoostingRegressor(max_depth=2, warm_start=True, random_state=42)

# 连续迭代五次的错误没有改善时，停止训练
min_val_error = float("inf")
error_going_up = 0
for n_estimators in range(1, 120):
    gbrt.n_estimators = n_estimators
    gbrt.fit(X_train, y_train)
    y_pred = gbrt.predict(X_val)
    val_error = mean_squared_error(y_val, y_pred)
    if val_error < min_val_error:
        min_val_error = val_error
        error_going_up = 0
    else:
        error_going_up += 1
        if error_going_up == 5:
            break  # early stopping
print(gbrt.n_estimators)
print("Minimum validation MSE:", min_val_error)

# %%
# XGBoost
#
import xgboost


# %%
# Stacking
# 上述的模型都是通过训练多个学习器后分别得到结果后整合为最终结果，整合的过程为投票、求平均、求加权
# 平均等统计方法。那为什么不把每个学习器得到的结果作为特征进行训练(Blend)，再预测出最后的结果，这就
# 是Stacking的思想

# %%
# 练习题 8
from scipy.io import loadmat
from sklearn.model_selection import train_test_split

mnist_path = "./datasets/mnist-original.mat"
mnist_raw = loadmat(mnist_path)
mnist = {
    "data": mnist_raw["data"].T,
    "target": mnist_raw["label"][0],
    "COL_NAMES": ["label", "data"],
    "DESCR": "mldata.org dataset: mnist-original",
}

X, y = mnist["data"], mnist["target"]
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=10000, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=10000, random_state=42)

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier

random_forest_clf = RandomForestClassifier(random_state=42, n_estimators=10)
extra_trees_clf = ExtraTreesClassifier(random_state=42)
svm_clf = LinearSVC(random_state=42, max_iter=2000)
mlp_clf = MLPClassifier(random_state=42)

estimators = [random_forest_clf, extra_trees_clf, svm_clf, mlp_clf]
for estimator in estimators:
    print("Training the", estimator)
    estimator.fit(X_train, y_train)

scores = [estimator.score(X_val, y_val) for estimator in estimators]
print(scores)

from sklearn.ensemble import VotingClassifier

named_estimators = [
    ("random_forest_clf", random_forest_clf),
    ("extra_trees_clf", extra_trees_clf),
    ("svm_clf", svm_clf),
    ("mlp_clf", mlp_clf),
]

voting_clf = VotingClassifier(named_estimators)
voting_clf.fit(X_train, y_train)
voting_clf.score(X_val, y_val)
scores = [estimator.score(X_val, y_val)
          for estimator in voting_clf.estimators_]
print(scores)

# %%
# 练习题 9
X_val_predictions = np.empty((len(X_val), len(estimators)), dtype=np.float32)
for index, estimator in enumerate(estimators):
    X_val_predictions[:, index] = estimator.predict(X_val)
print(X_val_predictions)

rnd_forest_blender = RandomForestClassifier(
    n_estimators=200, oob_score=True, random_state=42)
rnd_forest_blender.fit(X_val_predictions, y_val)
print(rnd_forest_blender.oob_score_)

X_test_predictions = np.empty((len(X_test), len(estimators)), dtype=np.float32)
for index, estimator in enumerate(estimators):
    X_test_predictions[:, index] = estimator.predict(X_test)
y_pred = rnd_forest_blender.predict(X_test_predictions) 

from sklearn.metrics import accuracy_score
score = accuracy_score(y_test, y_pred)
print(score)

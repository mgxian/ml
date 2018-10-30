# %%
# 读取数据
from IPython import display
from scipy.io import loadmat

mnist_path = "./datasets/mnist-original.mat"
mnist_raw = loadmat(mnist_path)
mnist = {
    "data": mnist_raw["data"].T,
    "target": mnist_raw["label"][0],
    "COL_NAMES": ["label", "data"],
    "DESCR": "mldata.org dataset: mnist-original",
}
# display.display(mnist)

X, y = mnist["data"], mnist["target"]
print(X.shape)

# %%
# 图形展示
import matplotlib
import matplotlib.pyplot as plt


def plot_digit(data):
    image = data.reshape(28, 28)
    plt.imshow(image, cmap=matplotlib.cm.binary,
               interpolation="nearest")
    plt.axis("off")


some_digit = X[36000]
plot_digit(some_digit)

# %%
# 拆分打乱数据
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
import numpy as np
shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

# %%
# 二分类
y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)

# 训练模型
from sklearn.linear_model import SGDClassifier
sgd_clf = SGDClassifier(max_iter=10, random_state=42)
sgd_clf.fit(X_train, y_train_5)

# 交叉验证
from sklearn.model_selection import cross_val_predict
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
print(y_train_pred)

# 查准率和查全率（Precision and Recall ）以及F1指标
from sklearn.metrics import precision_score, recall_score, f1_score
ps = precision_score(y_train_5, y_train_pred)
rs = recall_score(y_train_5, y_train_pred)
f1 = f1_score(y_train_5, y_train_pred)
print(ps, rs, f1)

# %%
# Precision/Recall 的权衡
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
y_scores = cross_val_predict(
    sgd_clf, X_train, y_train_5, cv=3, method="decision_function")
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)


def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="upper left")
    plt.ylim([0, 1])


plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.show()


def plot_precision_vs_recall(precisions, recalls):
    plt.plot(recalls, precisions, "b-", linewidth=2)
    plt.xlabel("Recall", fontsize=16)
    plt.ylabel("Precision", fontsize=16)
    plt.axis([0, 1, 0, 1])


plt.figure(figsize=(8, 6))
plot_precision_vs_recall(precisions, recalls)
plt.show()

from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)


def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)


plt.figure(figsize=(8, 6))
plot_roc_curve(fpr, tpr)
plt.show()

# %%
# 多分类
# 将二分类器扩展到多分类器一般有两种做法 OVO 和 OVA
from sklearn.linear_model import SGDClassifier
sgd_clf = SGDClassifier(max_iter=10, random_state=42)
sgd_clf.fit(X_train, y_train)
some_digit = X[36000]
sgd_clf.predict([some_digit])
# 第6个分数最高 所以数字为5
some_digit_scores = sgd_clf.decision_function([some_digit])
print(some_digit_scores)

# %%
# 评价分类器的好坏
from sklearn.model_selection import cross_val_score
score = cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy")
print(score)

# 采用预处理（标准化）来增加准确率
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
score = cross_val_score(sgd_clf, X_train_scaled,
                        y_train, cv=3, scoring="accuracy")
print(score)

# %%
# 错误分析
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)
conf_mx = confusion_matrix(y_train, y_train_pred)
# print(conf_mx)
plt.matshow(conf_mx, cmap=plt.cm.gray)
plt.show()
# 数字5比较暗，说明数字5被错分的比较多

# 去除正确的对角线的分布情况，只看错误的分布
row_sums = conf_mx.sum(axis=1, keepdims=True)
norm_conf_mx = conf_mx / row_sums.astype(np.float64)
# print(norm_conf_mx)
np.fill_diagonal(norm_conf_mx, 0)
plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
plt.show()


# %%
# 多标签分类
# 输出多个二值分类标签的就是多标签分类
# 不是所有的分类器都能进行多标签分类
# 评价多标签分类模型的方法可以对每种标签求F1值，再求平均值
from sklearn.neighbors import KNeighborsClassifier
y_train_large = (y_train >= 7)
y_train_odd = (y_train % 2 == 1)
y_multilabel = np.c_[y_train_large, y_train_odd]
knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_multilabel)
r = knn_clf.predict([some_digit])
print(r)

# %%
# 多输出分类 多输出分类是多标签分类更一般的一种形式
# 左图为的像素为特征 右图为标签
import numpy.random as rnd
noise1 = rnd.randint(0, 100, (len(X_train), 784))
noise2 = rnd.randint(0, 100, (len(X_test), 784))
X_train_mod = X_train + noise1
X_test_mod = X_test + noise2
y_train_mod = X_train
y_test_mod = X_test
import matplotlib.pyplot as plt
plt.subplot(1, 2, 1)
plt.imshow(X_train_mod[36000].reshape(28, 28), cmap=plt.cm.gray)
plt.subplot(1, 2, 2)
plt.imshow(X_train[36000].reshape(28, 28), cmap=plt.cm.gray)

# 训练一个KNN模型实现多输出分类（去噪）
from sklearn.neighbors import KNeighborsClassifier
knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train_mod, y_train_mod)
clean_digit = knn_clf.predict([X_train_mod[36000]])
plt.imshow(clean_digit.reshape(28, 28), cmap=plt.cm.gray)


# %%
# 练习题 1
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

knn_clf = KNeighborsClassifier()
param_grid = {
    'weights': ['uniform', 'distance'],
    'n_neighbors': [3, 4, 5],
}

grid_search = GridSearchCV(knn_clf, param_grid, cv=5, n_jobs=-1, verbose=3)
grid_search.fit(X_train, y_train)
print(grid_search.best_params_)
print(grid_search.best_score_)
y_pred = grid_search.predict(X_test)
score = accuracy_score(y_test, y_pred)
print(score)


# %%
# 练习题 2
from scipy.ndimage.interpolation import shift


def shift_image(image, dx, dy):
    image = image.reshape((28, 28))
    shifted_image = shift(image, [dx, dy], cval=0, mode="constant")
    return shifted_image.reshape([-1])


image = X_train[1000]
shifted_image_down = shift_image(image, 0, 5)
shifted_image_left = shift_image(image, -5, 0)

plt.figure(figsize=(12, 3))
plt.subplot(131)
plt.title("Original", fontsize=14)
plt.imshow(image.reshape(28, 28), interpolation="nearest", cmap="Greys")
plt.subplot(132)
plt.title("Shifted down", fontsize=14)
plt.imshow(shifted_image_down.reshape(28, 28),
           interpolation="nearest", cmap="Greys")
plt.subplot(133)
plt.title("Shifted left", fontsize=14)
plt.imshow(shifted_image_left.reshape(28, 28),
           interpolation="nearest", cmap="Greys")
plt.show()

X_train_augmented = [image for image in X_train]
y_train_augmented = [label for label in y_train]

for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
    for image, label in zip(X_train, y_train):
        X_train_augmented.append(shift_image(image, dx, dy))
        y_train_augmented.append(label)

X_train_augmented = np.array(X_train_augmented)
y_train_augmented = np.array(y_train_augmented)

shuffle_idx = np.random.permutation(len(X_train_augmented))
X_train_augmented = X_train_augmented[shuffle_idx]
y_train_augmented = y_train_augmented[shuffle_idx]
knn_clf = KNeighborsClassifier(**grid_search.best_params_)
knn_clf.fit(X_train_augmented, y_train_augmented)
y_pred = knn_clf.predict(X_test)
score = accuracy_score(y_test, y_pred)
print(score)


# %%
# 练习题 3
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

# 1. 加载数据
train_data = pd.read_csv('./datasets/titanic/train.csv')
test_data = pd.read_csv('./datasets/titanic/test.csv')


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names]


imputer = SimpleImputer(strategy='median')

num_pipeline = Pipeline([
    ('select_numeric', DataFrameSelector(["Age", "SibSp", "Parch", "Fare"])),
    ('imputer', imputer)
])
# num_pipeline.fit_transform(train_data)


class MostFrequentImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.most_frequent_ = pd.Series([X[c].value_counts().index[0] for c in X],
                                        index=X.columns)
        return self

    def transform(self, X, y=None):
        return X.fillna(self.most_frequent_)


cat_pipeline = Pipeline([
    ("select_cat", DataFrameSelector(["Pclass", "Sex", "Embarked"])),
    ("imputer", MostFrequentImputer()),
    ("cat_encoder", OneHotEncoder(sparse=False)),
])
# cat_pipeline.fit_transform(train_data)

preprocess_pipeline = FeatureUnion(transformer_list=[
    ("num_pipeline", num_pipeline),
    ("cat_pipeline", cat_pipeline),
])

# 2. 转换处理数据
X_train = preprocess_pipeline.fit_transform(train_data)
y_train = train_data["Survived"]

# 3. 训练模型
svm_clf = SVC(gamma='auto')
svm_clf.fit(X_train, y_train)

# 4. 预测
X_test = preprocess_pipeline.transform(test_data)
y_pred = svm_clf.predict(X_test)

# 5. 交叉验证评估模型
svm_scores = cross_val_score(svm_clf, X_train, y_train, cv=10)
print(svm_scores.mean())

# 6. 尝试其他模型
forest_clf = RandomForestClassifier(n_estimators=10, random_state=42)
forest_scores = cross_val_score(forest_clf, X_train, y_train, cv=10)
print(forest_scores.mean())

# 7. 可视化图示
plt.figure(figsize=(8, 4))
plt.plot([1]*10, svm_scores, ".")
plt.plot([2]*10, forest_scores, ".")
plt.boxplot([svm_scores, forest_scores], labels=("SVM","Random Forest"))
plt.ylabel("Accuracy", fontsize=14)
plt.show()
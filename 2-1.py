# %%
from IPython import display
import hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from pandas.plotting import scatter_matrix


# 1. 读取数据
housing = pd.read_csv('./datasets/housing/housing.csv')
# display.display(housing.head())
# # housing.hist(bins=50, figsize=(20,15))


# 2. 拆分数据
# def split_train_test(data, test_ratio):
#     shuffled_indices = np.random.permutation(len(data))
#     test_set_size = int(len(data) * test_ratio)
#     test_indices = shuffled_indices[:test_set_size]
#     train_indices = shuffled_indices[test_set_size:]
#     return data.iloc[train_indices], data.iloc[test_indices]


# def test_set_check(identifier, test_ratio, hash):
#     return hash(np.int64(identifier)).digest()[-1] < 256 * test_ratio


# def split_train_test_by_id(data, test_ratio, id_column, hash=hashlib.md5):
#     ids = data[id_column]
#     in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio, hash))
#     return data.loc[~in_test_set], data.loc[in_test_set]


# housing_with_id = housing.reset_index()  # adds an `index` column
# train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")
# train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
# print(len(train_set), "train +", len(test_set), "test")

housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

for set in (strat_train_set, strat_test_set):
    set.drop(["income_cat"], axis=1, inplace=True)

train_housing = strat_train_set.copy()

# 3. 图示化数据(可选)
# train_housing.plot(kind="scatter", x="longitude", y="latitude")
# train_housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
#                    s=housing["population"]/100, label="population",
#                    c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
#                    )
# plt.legend()

# 4. 查看数据相关性(可选)
# corr_matrix = train_housing.corr()
# corr_matrix["median_house_value"].sort_values(ascending=False)

# attributes = ["median_house_value", "median_income", "total_rooms"]
# scatter_matrix(housing[attributes], figsize=(12, 8))

# train_housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
# train_housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
# train_housing["population_per_household"]=housing["population"]/housing["households"]
# corr_matrix = train_housing.corr()
# corr_matrix["median_house_value"].sort_values(ascending=False)

# 5. 数据清洗
# 5.1 数据补值
train_housing = strat_train_set.drop("median_house_value", axis=1)
train_housing_labels = strat_train_set["median_house_value"].copy()

# sample_incomplete_rows = train_housing[train_housing.isnull().any(axis=1)].head()
# display.display(sample_incomplete_rows)

# housing_num = train_housing.drop("ocean_proximity", axis=1)

# train_housing.dropna(subset=["total_bedrooms"]) # option 1
# train_housing.drop("total_bedrooms", axis=1) # option 2
# median = train_housing["total_bedrooms"].median()
# train_housing["total_bedrooms"].fillna(median) # option 3

# from sklearn.impute import SimpleImputer
# imputer = SimpleImputer(strategy="median")
# imputer.fit_transform(housing_num)
# display.display(housing_num.head())

# 5.2 处理类别数据
# from sklearn.preprocessing import OrdinalEncoder
# ordinal_encoder = OrdinalEncoder()
# housing_cat = train_housing[["ocean_proximity"]]
# # display.display(housing_cat.head(10))
# housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
# print(housing_cat_encoded[:10])
# print(ordinal_encoder.categories_)

# from sklearn.preprocessing import OneHotEncoder
# cat_encoder = OneHotEncoder()
# housing_cat = train_housing[["ocean_proximity"]]
# housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
# print(housing_cat_1hot[:10].toarray())
# print(cat_encoder.categories_)

# 5.3 自定义处理数据
from sklearn.base import TransformerMixin, BaseEstimator

# column index
rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):  # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self  # nothing else to do

    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


# attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
# housing_extra_attribs = attr_adder.transform(housing.values)
# housing_extra_attribs = pd.DataFrame(
#     housing_extra_attribs,
#     columns=list(housing.columns)+["rooms_per_household", "population_per_household"])
# display.display(housing_extra_attribs.head())

# 5.4 特征缩放
# from sklearn.preprocessing import StandardScaler, MinMaxScaler
# scaler = MinMaxScaler()
# housing_age = train_housing[["housing_median_age"]]
# housing_age_scale = scaler.fit_transform(housing_age)
# print(housing_age_scale[:10])

# 5.5 转换管道
housing_num = train_housing.drop("ocean_proximity", axis=1)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler()),
])
# housing_num_tr = num_pipeline.fit_transform(housing_num)
# display.display(housing_num_tr[:5])

# 5.6 多特征转换管道
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(), cat_attribs),
])

housing_prepared = full_pipeline.fit_transform(train_housing)
# print(num_attribs)
# display.display(housing_prepared[:5])
# print(housing_prepared.shape)

# 6. 选择并训练模型

# 6.1 选择模型
# 线性回归模型
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, train_housing_labels)

# some test
# some_data = train_housing.iloc[:5]
# some_labels = train_housing_labels.iloc[:5]
# some_data_prepared = full_pipeline.transform(some_data)
# print("Predictions:", lin_reg.predict(some_data_prepared))
# print("Labels:", list(some_labels))

# 6.2 计算均方误差 RMSE
from sklearn.metrics import mean_squared_error, mean_absolute_error
housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(train_housing_labels, housing_predictions)
lin_mae = mean_absolute_error(train_housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
# print(lin_rmse, lin_mae)

# 决策树模型
from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(housing_prepared, train_housing_labels)

housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(train_housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
# print(tree_rmse)

# 随机森林
# from sklearn.ensemble import RandomForestRegressor
# forest_reg = RandomForestRegressor(n_estimators=10, random_state=42)
# forest_reg.fit(housing_prepared, train_housing_labels)
# housing_predictions = forest_reg.predict(housing_prepared)
# forest_mse = mean_squared_error(train_housing_labels, housing_predictions)
# forest_rmse = np.sqrt(forest_mse)
# print(forest_rmse)


# 6.3 交叉验证
from sklearn.model_selection import cross_val_score


def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())


scores = cross_val_score(tree_reg, housing_prepared,
                         train_housing_labels, scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)

lin_scores = cross_val_score(
    lin_reg, housing_prepared, train_housing_labels, scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)

# forest_scores = cross_val_score(
#     forest_reg, housing_prepared, train_housing_labels, scoring="neg_mean_squared_error", cv=10)
# forest_rmse_scores = np.sqrt(-forest_scores)

display_scores(lin_rmse_scores)
display_scores(tree_rmse_scores)
# display_scores(forest_rmse_scores)

# scores = cross_val_score(lin_reg, housing_prepared,
#                          train_housing_labels, scoring="neg_mean_squared_error", cv=10)
# pd.Series(np.sqrt(-scores)).describe()

# from sklearn.svm import SVR
# svm_reg = SVR(kernel="linear")
# svm_reg.fit(housing_prepared, train_housing_labels)
# housing_predictions = svm_reg.predict(housing_prepared)
# svm_mse = mean_squared_error(train_housing_labels, housing_predictions)
# svm_rmse = np.sqrt(svm_mse)
# print(svm_rmse)

# 6.4 模型调参

# 给定参数
# from sklearn.model_selection import GridSearchCV
# from sklearn.ensemble import RandomForestRegressor
# param_grid = [
#     # try 12 (3×4) combinations of hyperparameters
#     {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
#     # then try 6 (2×3) combinations with bootstrap set as False
#     {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
# ]

# forest_reg = RandomForestRegressor(random_state=42)
# # train across 5 folds, that's a total of (12+6)*5=90 rounds of training
# grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
#                            scoring='neg_mean_squared_error', return_train_score=True)
# grid_search.fit(housing_prepared, train_housing_labels)
# print(grid_search.best_params_)
# print(grid_search.best_estimator_)
# cvres = grid_search.cv_results_
# for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
#     print(np.sqrt(-mean_score), params)
# display.display(pd.DataFrame(grid_search.cv_results_))

# 随机
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import randint

param_distribs = {
    'n_estimators': randint(low=1, high=100),
    'max_features': randint(low=1, high=8),
}

forest_reg = RandomForestRegressor(random_state=42)
rnd_search = RandomizedSearchCV(forest_reg, param_distributions=param_distribs,
                                n_iter=10, cv=5, scoring='neg_mean_squared_error', random_state=42)
rnd_search.fit(housing_prepared, train_housing_labels)
cvres = rnd_search.cv_results_
# for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
#     print(np.sqrt(-mean_score), params)

# %%
# 查看特征对预测结果的贡献程度
feature_importances = rnd_search.best_estimator_.feature_importances_
print(feature_importances)
extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
cat_encoder = full_pipeline.named_transformers_["cat"]
cat_one_hot_attribs = list(cat_encoder.categories_[0])
attributes = num_attribs + extra_attribs + cat_one_hot_attribs
sorted_feature_importances = sorted(
    zip(feature_importances, attributes), reverse=True)
display.display(sorted_feature_importances)

# %%
# 使用最终模型在测试集上测试
final_model = rnd_search.best_estimator_

X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()

X_test_prepared = full_pipeline.transform(X_test)
final_predictions = final_model.predict(X_test_prepared)

final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
print('final_rmse: ', final_rmse)

# %%
# 计算测试集的 95% 部分 RMSE
from scipy import stats
confidence = 0.95
squared_errors = (final_predictions - y_test) ** 2
mean = squared_errors.mean()
m = len(squared_errors)
test_95_rmse = np.sqrt(stats.t.interval(
    confidence, m - 1, loc=np.mean(squared_errors), scale=stats.sem(squared_errors)))
print('test_95_rmse: ', test_95_rmse)

# %%
# 特征预处理转换和预测管道
full_pipeline_with_predictor = Pipeline([
    ("preparation", full_pipeline),
    ("linear", LinearRegression())
])
some_data = train_housing.iloc[:5]
some_labels = train_housing_labels.iloc[:5]
full_pipeline_with_predictor.fit(train_housing, train_housing_labels)
print(full_pipeline_with_predictor.predict(some_data))
print(some_labels.values)

# %%
# 模型持久化
my_model = full_pipeline_with_predictor
from sklearn.externals import joblib
joblib.dump(my_model, "my_model.pkl")
my_model_loaded = joblib.load("my_model.pkl")

# %%
# 可视化分布 RandomizedSearchCV
# rvs 产生服从指定分布的随机数
# expon 指数分布
# geom 几何分布
from scipy.stats import geom, expon
geom_distrib = geom(0.5).rvs(10000, random_state=42)
expon_distrib = expon(scale=1).rvs(10000, random_state=42)
plt.hist(geom_distrib, bins=50)
plt.show()
plt.hist(expon_distrib, bins=50)
plt.show()

# %%
# 练习题 1
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

param_grid = [
    {'kernel': ['linear'], 'C':[10., 30., 100.,
                                300., 1000., 3000., 10000., 30000.0]},
    {'kernel': ['rbf'], 'C':[1.0, 3.0, 10., 30., 100., 300.,
                             1000.0], 'gamma':[0.01, 0.03, 0.1, 0.3, 1.0, 3.0]},
]
svm_reg = SVR()
grid_search = GridSearchCV(svm_reg, param_grid=param_grid, cv=5,
                           scoring='neg_mean_squared_error', n_jobs=4, verbose=2)
grid_search.fit(housing_prepared, train_housing_labels)
negative_mse = grid_search.best_score_
print(grid_search.best_params_)
print(np.sqrt(-negative_mse))

# %%
# 练习题 2
# reciprocal 倒数
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import expon, reciprocal
param_distribs = {
    'kernel': ['linear', 'rbf'],
    'C': reciprocal(20, 200000),
    'gamma': expon(scale=1.0)
}
svm_reg = SVR()
# n_jobs 同时运行的job数
# n_iter 迭代次数
# cv 交叉验证的k值
rnd_search = RandomizedSearchCV(svm_reg, param_distributions=param_distribs,
                                n_iter=50, cv=5, scoring='neg_mean_squared_error',
                                verbose=2, n_jobs=4, random_state=42)
rnd_search.fit(housing_prepared, train_housing_labels)
negative_mse = grid_search.best_score_
print(grid_search.best_params_)
print(np.sqrt(-negative_mse))

# %%
from scipy.stats import expon
expon_distrib = expon(scale=1.)
samples = expon_distrib.rvs(10000, random_state=42)
plt.figure(figsize=(10, 4))
plt.subplot(121)
plt.title("Exponential distribution (scale=1.0)")
plt.hist(samples, bins=50)
plt.subplot(122)
plt.title("Log of this distribution")
plt.hist(np.log(samples), bins=50)
plt.show()
display.display(samples.max(), samples.min(), len(samples))

# %%
from scipy.stats import reciprocal
reciprocal_distrib = reciprocal(20, 200000)
samples = reciprocal_distrib.rvs(10000, random_state=42)
plt.figure(figsize=(10, 4))
plt.subplot(121)
plt.title("Reciprocal distribution (scale=1.0)")
plt.hist(samples, bins=50)
plt.subplot(122)
plt.title("Log of this distribution")
plt.hist(np.log(samples), bins=50)
plt.show()
display.display(samples.max(), samples.min(), len(samples))

# %%
# 练习题 3
from sklearn.base import TransformerMixin, BaseEstimator


def indices_of_top_k(arr, k):
    return np.sort(np.argpartition(np.array(arr), -k)[-k:])


class TopFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_importances, k):
        self.feature_importances = feature_importances
        self.k = k

    def fit(self, X, y=None):
        self.feature_indices_ = indices_of_top_k(
            self.feature_importances, self.k)
        return self

    def transform(self, X, y=None):
        return X[:, self.feature_indices_]


k = 5
top_k_feature_indices = indices_of_top_k(feature_importances, k)
# print(top_k_feature_indices)
preparation_and_feature_selection_pipeline = Pipeline([
    ('preparation', full_pipeline),
    ('feature_selection', TopFeatureSelector(feature_importances, k))
])

housing_prepared_top_k_features = preparation_and_feature_selection_pipeline.fit_transform(
    train_housing)
housing_prepared_top_k_features[0:3]
housing_prepared[0:3, top_k_feature_indices]

# %%
# 练习题 4
prepare_select_and_predict_pipeline = Pipeline([
    ("preparation", full_pipeline),
    ("feature_selection", TopFeatureSelector(feature_importances, k)),
    ("linear", LinearRegression())
])
some_data = train_housing.iloc[:5]
some_labels = train_housing_labels.iloc[:5]
prepare_select_and_predict_pipeline.fit(train_housing, train_housing_labels)
print(prepare_select_and_predict_pipeline.predict(some_data))
print(some_labels.values)

# %%
# 练习题 5
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR

param_grid = [
    {'preparation__num__imputer__strategy': ['mean', 'median', 'most_frequent'],
     'feature_selection__k': list(range(1, len(feature_importances) + 1))}
]

grid_search_prep = GridSearchCV(prepare_select_and_predict_pipeline,
                                param_grid, cv=5, scoring='neg_mean_squared_error',
                                verbose=2, n_jobs=4)
grid_search_prep.fit(train_housing, train_housing_labels)
print(grid_search_prep.best_params_)

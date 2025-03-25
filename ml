机器学习管道构建

‌场景‌：商品销量预测模型

pythonCopy Code
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor

# 1. 构建特征处理管道
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), ['price', 'rating']),
    ('cat', OneHotEncoder(), ['category', 'price_bin'])
])

# 2. 完整模型管道
pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('regressor', GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5
    ))
])

# 3. 交叉验证
scores = cross_val_score(pipeline, X_train, y_train, 
                        cv=5, scoring='neg_mean_squared_error')
print(f"RMSE平均得分: {np.sqrt(-scores.mean()):.2f}")

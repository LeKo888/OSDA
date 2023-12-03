import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import make_scorer, f1_score
from sklearn.metrics import accuracy_score

# 从 CSV 文件中读取数据
file_path = './boston_housing_data.csv'
data = pd.read_csv(file_path)

# 将房价转化为分类标签
data['Price_Category'] = pd.cut(data['PRICE'], bins=[0, data['PRICE'].median(), data['PRICE'].max()])

# 划分特征和目标变量
X = data.drop(['PRICE', 'Price_Category'], axis=1)
y = data['Price_Category']

# 将目标变量编码为数字
y = pd.factorize(y)[0]

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# 使用 XGBoost Classifier
xgb_classifier = XGBClassifier(random_state=42)

# 定义 F1 分数作为评估指标
f1_scorer = make_scorer(f1_score)
accuracy_scorer = make_scorer(accuracy_score)

# 交叉验证调整模型参数并评估 F1 分数
cv_scores = cross_val_score(xgb_classifier, X_train, y_train, cv=5, scoring=f1_scorer)

# 输出交叉验证的 F1 分数
print("Cross-Validation F1 Scores:", cv_scores)

# 输出平均 F1 分数
print("Average F1 Score:", np.mean(cv_scores))

# 训练模型
xgb_classifier.fit(X_train, y_train)

# 在数据集上进行预测
y_pred = xgb_classifier.predict(X_test)

# 输出最准确的分类结果
print("Most Accurate Predictions:")
print(pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}).head(10))

# 在数据集上进行预测
y_pred = xgb_classifier.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# _*_ coding:utf-8 _*_
# 为这个项目导入需要的库
import numpy as np
import pandas as pd
from time import time
from IPython.display import display # 允许为DataFrame使用display()
import matplotlib.pyplot as pl
import matplotlib.patches as mpatches

# 导入附加的可视化代码visuals.py
def evaluate(results, accuracy, f1):
    """
    Visualization code to display results of various learners.

    inputs:
      - learners: a list of supervised learners
      - stats: a list of dictionaries of the statistic results from 'train_predict()'
      - accuracy: The score for the naive predictor
      - f1: The score for the naive predictor
    """

    # Create figure
    fig, ax = pl.subplots(2, 3, figsize=(11, 7))

    # Constants
    bar_width = 0.1
    colors = ['#A00000', '#00A0A0', '#00A000', '#A0A000', '#0000A0', '#A000A0']

    # Super loop to plot four panels of data
    for k, learner in enumerate(results.keys()):
        for j, metric in enumerate(['train_time', 'acc_train', 'f_train', 'pred_time', 'acc_test', 'f_test']):
            for i in np.arange(3):
                # Creative plot code
                ax[j / 3, j % 3].bar(i + k * bar_width, results[learner][i][metric], width=bar_width, color=colors[k])
                ax[j / 3, j % 3].set_xticks([0.45, 1.45, 2.45])
                ax[j / 3, j % 3].set_xticklabels(["1%", "10%", "100%"])
                ax[j / 3, j % 3].set_xlabel("Training Set Size")
                ax[j / 3, j % 3].set_xlim((-0.1, 3.0))

    # Add unique y-labels
    ax[0, 0].set_ylabel("Time (in seconds)")
    ax[0, 1].set_ylabel("Accuracy Score")
    ax[0, 2].set_ylabel("F-score")
    ax[1, 0].set_ylabel("Time (in seconds)")
    ax[1, 1].set_ylabel("Accuracy Score")
    ax[1, 2].set_ylabel("F-score")

    # Add titles
    ax[0, 0].set_title("Model Training")
    ax[0, 1].set_title("Accuracy Score on Training Subset")
    ax[0, 2].set_title("F-score on Training Subset")
    ax[1, 0].set_title("Model Predicting")
    ax[1, 1].set_title("Accuracy Score on Testing Set")
    ax[1, 2].set_title("F-score on Testing Set")

    # Add horizontal lines for naive predictors
    ax[0, 1].axhline(y=accuracy, xmin=-0.1, xmax=3.0, linewidth=1, color='k', linestyle='dashed')
    ax[1, 1].axhline(y=accuracy, xmin=-0.1, xmax=3.0, linewidth=1, color='k', linestyle='dashed')
    ax[0, 2].axhline(y=f1, xmin=-0.1, xmax=3.0, linewidth=1, color='k', linestyle='dashed')
    ax[1, 2].axhline(y=f1, xmin=-0.1, xmax=3.0, linewidth=1, color='k', linestyle='dashed')

    # Set y-limits for score panels
    ax[0, 1].set_ylim((0, 1))
    ax[0, 2].set_ylim((0, 1))
    ax[1, 1].set_ylim((0, 1))
    ax[1, 2].set_ylim((0, 1))

    # Create patches for the legend
    patches = []
    for i, learner in enumerate(results.keys()):
        patches.append(mpatches.Patch(color=colors[i], label=learner))
    pl.legend(handles=patches, bbox_to_anchor=(-.80, 2.53), \
              loc='upper center', borderaxespad=0., ncol=6, fontsize='x-large')

    # Aesthetics
    pl.suptitle("Performance Metrics for Three Supervised Learning Models", fontsize=16, y=1.10)
    pl.tight_layout()
    pl.show()

def feature_plot(importances, X_train, y_train):
    # Display the five most important features
    indices = np.argsort(importances)[::-1]
    columns = X_train.columns.values[indices[:5]]
    values = importances[indices][:5]

    # Creat the plot
    fig = pl.figure(figsize=(9, 5))
    pl.title("Normalized Weights for First Five Most Predictive Features", fontsize=16)
    pl.bar(np.arange(5), values, width=0.6, align="center", color='#00A000', \
           label="Feature Weight")
    pl.bar(np.arange(5) - 0.3, np.cumsum(values), width=0.2, align="center", color='#00A0A0', \
           label="Cumulative Feature Weight")
    pl.xticks(np.arange(5), columns)
    pl.xlim((-0.5, 4.5))
    pl.ylabel("Weight", fontsize=12)
    pl.xlabel("Feature", fontsize=12)

    pl.legend(loc='upper center')
    pl.tight_layout()
    pl.show()

# 为notebook提供更加漂亮的可视化
#%matplotlib inline

# 导入人口普查数据
data = pd.read_csv("census.csv")

# 成功 - 显示第一条记录
display(data.head(n=1))

# TODO：总的记录数
n_records = len(data.index)

# TODO：被调查者的收入大于$50,000的人数
n_greater_50k = len(data[data["income"]=='>50K'])

# TODO：被调查者的收入最多为$50,000的人数
n_at_most_50k = len(data[data["income"]=='<=50K'])

# TODO：被调查者收入大于$50,000所占的比例
greater_percent = 100*n_greater_50k/n_records

# 打印结果
print "Total number of records: {}".format(n_records)
print "Individuals making more than $50,000: {}".format(n_greater_50k)
print "Individuals making at most $50,000: {}".format(n_at_most_50k)
print "Percentage of individuals making more than $50,000: {:.2f}%".format(greater_percent)

# 将数据切分成特征和对应的标签
income_raw = data['income']
features_raw = data.drop('income', axis = 1)

# 对于倾斜的数据使用Log转换
skewed = ['capital-gain', 'capital-loss']
features_raw[skewed] = data[skewed].apply(lambda x: np.log(x + 1))

# 导入sklearn.preprocessing.StandardScaler
from sklearn.preprocessing import MinMaxScaler

# 初始化一个 scaler，并将它施加到特征上
scaler = MinMaxScaler()
numerical = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
features_raw[numerical] = scaler.fit_transform(data[numerical])

# 显示一个经过缩放的样例记录
#display(features_raw.head(n = 1))

# TODO：使用pandas.get_dummies()对'features_raw'数据进行独热编码
features = pd.get_dummies(features_raw)

# TODO：将'income_raw'编码成数字值
income = (pd.get_dummies(data['income']))[">50K"]

# 打印经过独热编码之后的特征数量
encoded = list(features.columns)
print "{} total features after one-hot encoding.".format(len(encoded))

# 移除下面一行的注释以观察编码的特征名字
#print encoded

# 导入 train_test_split
from sklearn.model_selection import train_test_split

# 将'features'和'income'数据切分成训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(features, income, test_size = 0.2, random_state = 0)

# 显示切分的结果
print "Training set has {} samples.".format(X_train.shape[0])
print "Testing set has {} samples.".format(X_test.shape[0])

# TODO： 计算准确率
accuracy = float(n_greater_50k)/n_records

# TODO： 使用上面的公式，并设置beta=0.5计算F-score
beta = 0.5
precision = 1.0
recall = float(n_greater_50k)/n_records
fscore = (1+beta**2)*(precision*recall)/((beta**2)*precision + recall)

# 打印结果
print "Naive Predictor: [Accuracy score: {:.4f}, F-score: {:.4f}]".format(accuracy, fscore)

from sklearn.metrics import fbeta_score, accuracy_score

def train_predict(learner, sample_size, X_train, y_train, X_test, y_test):
    '''
    inputs:
       - learner: the learning algorithm to be trained and predicted on
       - sample_size: the size of samples (number) to be drawn from training set
       - X_train: features training set
       - y_train: income training set
       - X_test: features testing set
       - y_test: income testing set
    '''

    results = {}

    # TODO：使用sample_size大小的训练数据来拟合学习器
    # TODO: Fit the learner to the training data using slicing with 'sample_size'
    start = time()  # 获得程序开始时间
    learner.fit(X_train[:sample_size], y_train[:sample_size])
    end = time()  # 获得程序结束时间

    # TODO：计算训练时间
    results['train_time'] = end - start

    # TODO: 得到在测试集上的预测值
    #       然后得到对前300个训练数据的预测结果
    start = time()  # 获得程序开始时间
    predictions_test = learner.predict(X_test)
    predictions_train = learner.predict(X_train[:300])
    end = time()  # 获得程序结束时间

    # TODO：计算预测用时
    results['pred_time'] = end - start

    # TODO：计算在最前面的300个训练数据的准确率
    results['acc_train'] = accuracy_score(y_train[:300], predictions_train)

    # TODO：计算在测试集上的准确率
    results['acc_test'] = accuracy_score(y_test, predictions_test)

    # TODO：计算在最前面300个训练数据上的F-score
    results['f_train'] = fbeta_score(y_train[:300], predictions_train, 0.5)

    # TODO：计算测试集上的F-score
    results['f_test'] = fbeta_score(y_test, predictions_test, 0.5)

    # 成功
    print "{} trained on {} samples.".format(learner.__class__.__name__, sample_size)

    # 返回结果
    return results


# TODO：从sklearn中导入三个监督学习模型
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn import linear_model
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

# # TODO：初始化三个模型
# #clf_A = KNeighborsClassifier(n_neighbors=3)
# # clf_B = SVC(kernel='rbf')
# # clf_C = linear_model.SGDClassifier()
# clf_B = svm.SVC(kernel='rbf')
# #clf_B = linear_model.SGDClassifier()
# clf_A = DecisionTreeClassifier()
# #clf_C = BaggingClassifier()
# clf_C = GaussianNB()
# clf_D = AdaBoostClassifier()
# clf_E = RandomForestClassifier()
# clf_F = GradientBoostingClassifier()
#
#
# # TODO：计算1%， 10%， 100%的训练数据分别对应多少点
# samples_1 = len(y_train) / 100
# samples_10 = len(y_train) / 10
# samples_100 = len(y_train) / 4
#
# # # 收集学习器的结果
# results = {}
# for clf in [clf_A, clf_B, clf_C, clf_D, clf_E, clf_F]:
#     clf_name = clf.__class__.__name__
#     results[clf_name] = {}
#     for i, samples in enumerate([samples_1, samples_10, samples_100]):
#         results[clf_name][i] = \
#         train_predict(clf, samples, X_train, y_train, X_test, y_test)
#
# # 对选择的三个模型得到的评价结果进行可视化
# evaluate(results, accuracy, fscore)

# # TODO：导入'GridSearchCV', 'make_scorer'和其他一些需要的库
# from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import make_scorer
#
# # TODO：初始化分类器
# clf = DecisionTreeClassifier()
#
# # TODO：创建你希望调节的参数列表
# #parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
# #parameters = {'algorithm':('SAMME','SAMME.R'), 'learning_rate': [1,5,10], 'n_estimators': [40,50,60]}
# parameters = {'max_depth':range(4,20,2),'class_weight':('balanced',None),'min_samples_split':[2,3,4]}
#
# # TODO：创建一个fbeta_score打分对象
# scorer = make_scorer(fbeta_score, beta=0.5)
#
# # TODO：在分类器上使用网格搜索，使用'scorer'作为评价函数
# grid_obj = GridSearchCV(clf, parameters,scoring=scorer)
#
# # TODO：用训练数据拟合网格搜索对象并找到最佳参数
# grid_obj = grid_obj.fit(X_train, y_train)
#
# # 得到estimator
# best_clf = grid_obj.best_estimator_
#
# # 使用没有调优的模型做预测
# predictions = (clf.fit(X_train, y_train)).predict(X_test)
# best_predictions = best_clf.predict(X_test)
#
# # 汇报调参前和调参后的分数
# print "Unoptimized model\n------"
# print "Accuracy score on testing data: {:.4f}".format(accuracy_score(y_test, predictions))
# print "F-score on testing data: {:.4f}".format(fbeta_score(y_test, predictions, beta = 0.5))
# print "\nOptimized Model\n------"
# print "Final accuracy score on the testing data: {:.4f}".format(accuracy_score(y_test, best_predictions))
# print "Final F-score on the testing data: {:.4f}".format(fbeta_score(y_test, best_predictions, beta = 0.5))
# print "Final Parameter 'max_depth' is:"
# print best_clf.get_params()

# TODO：导入一个有'feature_importances_'的监督学习模型
# from sklearn.ensemble import AdaBoostClassifier
# clf = AdaBoostClassifier()
#
# # TODO：在训练集上训练一个监督学习模型
# model = clf.fit(X_train, y_train)
#
# # TODO： 提取特征重要性
# importances = model.feature_importances_
#
# # 绘图
# feature_plot(importances, X_train, y_train)
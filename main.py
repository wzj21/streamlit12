# 导入必要的库
import pickle
import pandas as pd
import numpy as np
import streamlit as st

# 设置streamlit页面配置
st.set_page_config(layout="wide", page_title="脓毒症合并低蛋白血症的预后模型")

# 在侧边栏显示模型推荐和特征解释的警告
st.sidebar.warning('其余维度变量是ICU入院后的首24小时内的数值。')
st.sidebar.warning('对于crrt，0代表无疾病; 1表示有疾病。')

# 显示预后模型的信息
st.info('#### 脓毒症合并低蛋白血症的预后模型')

# 指定要使用的模型文件
model_file = "Stacking.pkl"  # 在这里使用你的模型文件

# 加载并转置特征范围excel文件
val = pd.read_excel("变量范围.xlsx").T
val.fillna("", inplace=True)

# 从模型文件中加载选定的模型
with open(model_file, 'rb') as fmodel:
    model = pickle.load(fmodel)

# 在Streamlit界面中设置参数输入的展开器
with st.expander("**Params setting**", True):
    col = st.columns(5)
    k = 0
    # 遍历所有特征并为每个特征创建一个输入字段
    for i in range(1, val.shape[0]):
        if val.iloc[i][3] !="" and val.index[i] in model.feature_names_in_:
            st.session_state[val.index[i]] = col[k%5].number_input(val.index[i].replace("_", " ")+"("+val.iloc[i][3]+")", min_value=float(val.iloc[i][0]), max_value=float(val.iloc[i][1]), step=float(val.iloc[i][2]))
            k = k+1
        elif val.index[i] in model.feature_names_in_:
            st.session_state[val.index[i]] = col[k%5].number_input(val.index[i].replace("_", " "), min_value=float(val.iloc[i][0]), max_value=float(val.iloc[i][1]), step=float(val.iloc[i][2]))
            k = k+1

    col1 = st.columns(5)
    # 添加“开始预测”按钮
    start = col1[2].button("Start predict", use_container_width=True)

# 如果点击了“开始预测”按钮，使用输入的特征值进行预测
if start:
    X = np.array([[st.session_state[i] for i in model.feature_names_in_]])

    with st.expander("**Current parameters and predict result**", True):
        # 显示输入的特征值
        p = pd.DataFrame([{i:st.session_state[i] for i in model.feature_names_in_}])
        p.index = ["params"]
        st.write(p)

        # 预测结果并显示
        y_pred = model.predict(X)
        y_pred_prob = model.predict_proba(X)
        res = 'Non-Survival' if y_pred[0] else 'survival'
        st.success(f"预测成功。结果是 **{res}**，概率如下：")

        # 显示预测的概率
        pred_prob = pd.DataFrame([[round(y_pred_prob[0][0], 3), round(y_pred_prob[0][1], 3)]], columns=['Survival','Non-survival'])
        pred_prob.index = ["pred prob"]
        st.dataframe(pred_prob, use_container_width=True)

# 如果没有点击“开始预测”按钮，显示警告
else:
    with st.expander("**Current parameters and predict result**", True):
        st.warning("**当前没有使用模型进行预测！**")

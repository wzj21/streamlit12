import pickle
import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(layout="wide", page_title="Prognostic model for sepsis combined with hypoproteinemia")


st.sidebar.warning('The remaining dimension variables are values within the first 24 hours after admission to the ICU')
st.sidebar.warning('For CRRT, 0 represents no disease; 1 indicates the presence of disease')


st.info('#### Prognostic model for sepsis combined with hypoproteinemia')


model_file = "Stacking.pkl"  # 在这里使用你的模型文件


val = pd.read_excel("变量范围.xlsx").T
val.fillna("", inplace=True)


with open(model_file, 'rb') as fmodel:
    model = pickle.load(fmodel)


with st.expander("**Params setting**", True):
    col = st.columns(5)
    k = 0

    for i in range(1, val.shape[0]):
        if val.iloc[i][3] !="" and val.index[i] in model.feature_names_in_:
            st.session_state[val.index[i]] = col[k%5].number_input(val.index[i].replace("_", " ")+"("+val.iloc[i][3]+")", min_value=float(val.iloc[i][0]), max_value=float(val.iloc[i][1]), step=float(val.iloc[i][2]))
            k = k+1
        elif val.index[i] in model.feature_names_in_:
            st.session_state[val.index[i]] = col[k%5].number_input(val.index[i].replace("_", " "), min_value=float(val.iloc[i][0]), max_value=float(val.iloc[i][1]), step=float(val.iloc[i][2]))
            k = k+1

    col1 = st.columns(5)

    start = col1[2].button("Start predict", use_container_width=True)

if start:
    X = np.array([[st.session_state[i] for i in model.feature_names_in_]])

    with st.expander("**Current parameters and predict result**", True):

        p = pd.DataFrame([{i:st.session_state[i] for i in model.feature_names_in_}])
        p.index = ["params"]
        st.write(p)


        y_pred = model.predict(X)
        y_pred_prob = model.predict_proba(X)
        res = 'Non-Survival' if y_pred[0] else 'survival'
        st.success(f"Prediction successful. The result is {res}, with probabilities as follows:")


        pred_prob = pd.DataFrame([[round(y_pred_prob[0][0], 3), round(y_pred_prob[0][1], 3)]], columns=['Survival','Non-survival'])
        pred_prob.index = ["pred prob"]
        st.dataframe(pred_prob, use_container_width=True)


else:
    with st.expander("**Current parameters and predict result**", True):


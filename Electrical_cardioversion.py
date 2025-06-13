import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the trained model
model = joblib.load('model.pkl')  # 加载训练好的ET模型

# Define the feature options for AtrialFibrillationType
af_type_options = {
    1: 'Paroxysmal (1)',  # 发作性
    2: 'Persistent (2)',  # 持续性
    3: 'Permanent (3)'    # 持久性
}

# Streamlit UI
st.title("Electrical Cardioversion Predictor")  # 电复律预测器

# Sidebar for input options
st.sidebar.header("Input Sample Data")  # 侧边栏输入样本数据

# NtproBNP input
ntprobnp = st.sidebar.number_input("NtproBNP:", min_value=0, max_value=10000, value=1000)  # NtproBNP输入框

# BMI input
bmi = st.sidebar.number_input("BMI:", min_value=10, max_value=60, value=25)  # BMI输入框

# LeftAtrialDiam input
leftatrialdiam = st.sidebar.number_input("Left Atrial Diameter (LeftAtrialDiam):", min_value=0, max_value=100, value=40)  # 左房直径输入框

# AFCourse input
afcourse = st.sidebar.number_input("Atrial Fibrillation Course (AFCourse):", min_value=0, max_value=10, value=3)  # 心房颤动病程输入框

# AtrialFibrillationType input
af_type = st.sidebar.selectbox("Atrial Fibrillation Type:", options=list(af_type_options.keys()), format_func=lambda x: af_type_options[x])  # 心房颤动类型选择框

# Systolic BP input
systolicbp = st.sidebar.number_input("Systolic Blood Pressure (SystolicBP):", min_value=50, max_value=200, value=120)  # 收缩压输入框

# Age input
age = st.sidebar.number_input("Age:", min_value=1, max_value=120, value=50)  # 年龄输入框

# AST input
ast = st.sidebar.number_input("AST:", min_value=0, max_value=200, value=25)  # AST输入框

# Process the input and make a prediction
feature_values = [ntprobnp, bmi, leftatrialdiam, afcourse, af_type, systolicbp, age, ast]  # 收集所有输入的特征
features = np.array([feature_values])  # 转换为NumPy数组

if st.button("Make Prediction"):  # 如果点击了预测按钮
    # Predict the class and probabilities
    predicted_class = model.predict(features)[0]  # 预测电复律结果
    predicted_proba = model.predict_proba(features)[0]  # 预测各类别的概率

    # Display the prediction results
    st.write(f"**Predicted Class (0 = No, 1 = Yes):** {predicted_class}")  # 显示预测类别
    st.write(f"**Prediction Probabilities:** {predicted_proba}")  # 显示各类别的预测概率

    # Generate advice based on the prediction result
    probability = predicted_proba[predicted_class] * 100  # 根据预测类别获取对应的概率，并转化为百分比

    if predicted_class == 1:  # 如果预测为电复律治疗
        advice = (
            f"According to our model, you may require electrical cardioversion. "
            f"The probability of needing electrical cardioversion is {probability:.1f}%. "
            "This suggests that you may have a higher risk of requiring this treatment. "
            "I recommend consulting with a cardiologist for further examination and possible treatment options."
        )  # 如果预测为需要电复律，给出相关建议
    else:  # 如果预测为不需要电复律
        advice = (
            f"According to our model, you do not require electrical cardioversion. "
            f"The probability of not needing electrical cardioversion is {probability:.1f}%. "
            "However, it is still important to continue regular monitoring of your heart health. "
            "Please ensure you maintain a healthy lifestyle and seek medical attention if needed."
        )  # 如果预测为不需要电复律，给出相关建议

    st.write(advice)  # 显示建议

    # Visualize the prediction probabilities
    sample_prob = {
        'No Electrical Cardioversion': predicted_proba[0],  # 不需要电复律的概率
        'Needs Electrical Cardioversion': predicted_proba[1]  # 需要电复律的概率
    }

    # Set figure size
    plt.figure(figsize=(10, 3))  # 设置图形大小

    # Create bar chart
    bars = plt.barh(['No Electrical Cardioversion', 'Needs Electrical Cardioversion'], 
                    [sample_prob['No Electrical Cardioversion'], sample_prob['Needs Electrical Cardioversion']], 
                    color=['#4caf50', '#fe346e'])  # 绘制水平条形图

    # Add title and labels, set font bold and increase font size
    plt.title("Prediction Probability for Electrical Cardioversion", fontsize=20, fontweight='bold')  # 添加图表标题，并设置字体大小和加粗
    plt.xlabel("Probability", fontsize=14, fontweight='bold')  # 添加X轴标签，并设置字体大小和加粗
    plt.ylabel("Classes", fontsize=14, fontweight='bold')  # 添加Y轴标签，并设置字体大小和加粗

    # Add probability text labels, adjust position to avoid overlap, set font bold
    for i, v in enumerate([sample_prob['No Electrical Cardioversion'], sample_prob['Needs Electrical Cardioversion']]):  # 为每个条形图添加概率文本标签
        plt.text(v + 0.0001, i, f"{v:.2f}", va='center', fontsize=14, color='black', fontweight='bold')  # 设置标签位置、字体加粗

    # Hide other axes (top, right, bottom)
    plt.gca().spines['top'].set_visible(False)  # 隐藏顶部边框
    plt.gca().spines['right'].set_visible(False)  # 隐藏右边框

    # Show the plot
    st.pyplot(plt)  # 显示图表

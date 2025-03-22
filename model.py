import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import recall_score , precision_score , f1_score , confusion_matrix , accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt




st.title("AndroidMalware detection using decision tree")

uploaded_data=st.file_uploader("upload your dataset" , type=["csv"])

if uploaded_data is not None:
    data=pd.read_csv(uploaded_data)
    st.write(data.head())
    
    
    x=data.drop('Result' , axis= 1)
    y=data['Result']
    
    
    x_train , x_test , y_train , y_test= train_test_split(x , y , test_size=0.2 , random_state=42)
    
    dt_classifier=DecisionTreeClassifier()
    dt_classifier.fit(x_train , y_train)
    
    
    y_pred=dt_classifier.predict(x_test)
    
    
    #metrics
    precision=precision_score(y_test , y_pred)
    recall=recall_score(y_test , y_pred)
    accuracy=accuracy_score(y_test , y_pred)
    f1=f1_score(y_test , y_pred)
    conf=confusion_matrix(y_test , y_pred)
    
    
    #display
    st.subheader("model performance")
    st.write(f"accuracy : {accuracy :.2f}")
    st.write(f"precision : {precision :.2f}")
    st.write(f"recall : {recall :.2f}")
    st.write(f"f1score : {f1 :.2f}")
    
    
    #confusion metrics 
    st.subheader("confusion matrix")
    fig,ax=plt.subplots()
    sns.heatmap(conf , annot=True , cmap='Blues' , fmt="d" , ax=ax)
    st.pyplot(fig)
    
    #model performance 
    st.subheader("Model performance")
    fig,ax=plt.subplots()
    ax.bar(['precision ', 'recall' ,' f1 score'   ] , [precision , recall , f1])
    ax.set_ylabel("value")
    st.pyplot(fig)
    
    
    st.subheader("Predict malware type")
    input_data=[]  #values for features 
    for col in x.columns:
        value=st.number_input(f"Enter {col}" , min_value=0.0 , format="%.2f")
        input_data.append(value)
        
        
    #make prediction
    prediction=None 
    if st.button("predict"):
        input_df=pd.DataFrame([input_data] , columns=x.columns)
        prediction=dt_classifier.predict(input_df)[0]   
        
        
        if prediction==1:
            st.error("Malware detected")
        else:
            st.success("safe application")
            
        
    
    
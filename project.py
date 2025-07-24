import streamlit as st
import pandas as pd
from numpy import mean
from sklearn.model_selection import train_test_split
#for graph ploting
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
#for oversampling
from imblearn.combine import SMOTEENN
#for algorithm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from catboost import CatBoostClassifier
#for metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score,KFold

# Define the app title and description
st.title("Machine Learning Classifier Performance")
st.write("This app allows you to upload a CSV file, select a machine learning algorithm, and see the algorithm's performance.")

# Create file uploader
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

# Create algorithm selector
algorithm = st.selectbox(
    "Select a machine learning algorithm",
    ("Choose the algorithm","k-Nearest Neigbours (kNN)", "Naive Bayes (NB)", "Support Vector Machine (SVM)", "Multilayer Perceptron (MLP)", "Extreme Gradient Boost (XGBoost)", "CatBoost", "Compare All Algorithms","K_Fold")
)
targetBal = st.selectbox(
    "Choose if you want to balance the target variable",
    ("No (Base paper implementation)","Yes")
)

submit_btn = st.button("Submit")

box_plot_btn = st.sidebar.button("Box Plot")
violin_plot_btn = st.sidebar.button("Violin Plot")
pair_plot_btn = st.sidebar.button("Pair Plot")
heatmap_btn = st.sidebar.button("Heat Map")
gender_btn = st.sidebar.button('Gender Visualization')
notch_btn = st.sidebar.button('Notched Box plot')

def data_reading():
    # Load data into a Pandas DataFrame
    data = pd.read_csv(uploaded_file)
    data["target"] = data["target"].apply(lambda x: 0 if x==0 else 1)
    # Separate features and target variable
    X = data.drop("target", axis=1)
    Y = data["target"]
    # Split the data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    scaler = StandardScaler()
    X_train=scaler.fit_transform(X_train)
    X_test=scaler.transform(X_test)
    return X_train,X_test,Y_train,Y_test

def oversampled_data_reading():
    data = pd.read_csv(uploaded_file)
    data["target"] = data["target"].apply(lambda x: 0 if x==0 else 1)
    # Separate features and target variable
    X = data.drop('target', axis=1)
    Y = data['target']
    # Combine SMOTE and ENN to balance the dataset
    smote_enn = SMOTEENN(random_state=42)
    X_balanced, Y_balanced = smote_enn.fit_resample(X, Y)

    # Split the data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X_balanced, Y_balanced, test_size=0.2, random_state=42)

    # Perform oversampling of the minority class using SMOTE
    #smote = SMOTE(random_state=42)
    #X_train_resampled, Y_train_resampled = smote.fit_resample(X_train, Y_train)

    X_train_resampled, Y_train_resampled  = X_train,Y_train
            
    # Perform standard scaling on the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_resampled)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled,X_test_scaled,Y_train_resampled,X_test,Y_test

def box_plot():
    # Load sample data
    df = pd.read_csv(uploaded_file)
    df["target"] = df["target"].apply(lambda x: 0 if x==0 else 1)
    # Create a box plot using Seaborn
    fig = plt.figure(figsize=(10,10))
    st.write("Box Plot")
    sns.set_style('whitegrid')
    sns.boxplot(data=df, x="target", y="slope")
    st.pyplot(fig)

def violin_plot():
    df = pd.read_csv(uploaded_file)
    df["target"] = df["target"].apply(lambda x: 0 if x==0 else 1)
    # Create a box plot using Seaborn
    fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True, figsize=(10, 6))
    st.write("Violin Plot and Strip plot")
    sns.set_style('whitegrid')
    sns.stripplot(data=df, x="target", y="chol", jitter=True, size=4,ax=ax1)
    sns.violinplot(data=df, x="target", y="chol",ax=ax2)
    ax2.tick_params(left=False)
    st.pyplot(fig)

def pair_plot():
    data1 = pd.read_csv(uploaded_file)
    data1["target"] = data1["target"].apply(lambda x:0 if x==0 else 1)
    # data1 = data1.drop('target',axis = 1)
    st.write("Pair Plot")
    sns.set(font_scale=2)
    sns.set(rc={'figure.figsize':(10,10)})
    sns_plot = sns.pairplot(data1,diag_kind="kde", hue="target")
    st.pyplot(sns_plot.fig)

def heatmap():
    data = pd.read_csv(uploaded_file)
    data["target"] = data["target"].apply(lambda x: 0 if x==0 else 1)
    # Separate features and target variable
    X = data.drop("target", axis=1)
    Y = data["target"]
    # Split the data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    
    corr_matrix = X_train.corr()
    # corr_matrix
    st.write("Correlation Matrix")
    fig,ax = plt.subplots(figsize=(10,10))
    sns.heatmap(corr_matrix,annot=True, cmap='coolwarm',annot_kws={"size":12},ax=ax)
    st.pyplot(fig)

def gender_plot():
         # Load sample data
    df = pd.read_csv(uploaded_file)
    df["target"] = df["target"].apply(lambda x: 0 if x==0 else 1)
    fig = plt.figure(figsize=(10,10))
    sns.set_style('whitegrid')
    colors = ['#98FB98','#FFFF99']
    # Group the data by gender and count the number of occurrences
    gender_counts = df.groupby("Sex")["Sex"].count()

    # Create a donut chart
    fig, ax = plt.subplots(figsize=(7,7))
    ax.pie(gender_counts.values, autopct="%1.1f%%", startangle=180, counterclock=False, pctdistance=0.8, wedgeprops=dict(width=0.5),colors=colors)
    ax.add_artist(plt.Circle((0, 0), 0.6, fc="white"))
    plt.title("Gender")

    labels = ['Female', 'Male']
    plt.legend(labels, loc="best", bbox_to_anchor=(1, 0.5))
    st.pyplot(fig)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import matplotlib.patches as mpatches

def notched_plot():
    df = pd.read_csv(uploaded_file)
    df["target"] = df["target"].apply(lambda x: 0 if x == 0 else 1)

    fig, ax = plt.subplots(figsize=(28, 6))  # Capture the figure and axis

    colors = ['#98FB98', '#FFFF99']

    # Create a horizontal notched boxplot with whiskers for target
    sns.boxplot(
        x="Age", 
        y="target", 
        data=df, 
        orient="h", 
        notch=True, 
        whis=1.5, 
        palette=colors,
        order=[1, 0],
        ax=ax  # use the axis object we created
    )

    # Set the title and labels for the plot
    ax.set_title("Notched Boxplot for Age vs Target", fontsize=50)
    ax.set_xlabel("Target", fontsize=50)
    ax.set_ylabel("")
    ax.tick_params(axis='x', labelsize=50)
    ax.tick_params(axis='y', labelsize=50)

    legend_handles = [
        mpatches.Patch(color=colors[0], label="0"),
        mpatches.Patch(color=colors[1], label="1")
    ]

    # Add the legend
    ax.legend(handles=legend_handles, loc="best", bbox_to_anchor=(1, 1))

    # Display the plot in Streamlit
    st.pyplot(fig)

if violin_plot_btn:
    if uploaded_file is not None:
        violin_plot()
    else:
        st.write("Uploade File")

if pair_plot_btn:
    if uploaded_file is not None:
        pair_plot()
    else:
        st.write("Uploade File")

if heatmap_btn:
    if uploaded_file is not None:
        heatmap()
    else:
        st.write("Uploade File")

if box_plot_btn:
    if uploaded_file is not None:
        box_plot()
    else:
        st.write("Uploade File")
if gender_btn:
    if uploaded_file is not None:
        gender_plot()
    else:
        st.write("Upload File")
if notch_btn:
    if uploaded_file is not None:
        notched_plot()
    else:
        st.write("Upload File")

# Define the function to train and test the selected algorithm
def train_test_model(algorithm, X_train, X_test, Y_train, Y_test):
    if algorithm == "k-Nearest Neigbours (kNN)":
        model = KNeighborsClassifier(n_neighbors=8)
        model.fit(X_train, Y_train)
    elif algorithm == "Naive Bayes (NB)":
        model = GaussianNB()
        model.fit(X_train, Y_train)
    elif algorithm == "Support Vector Machine (SVM)":
        model = SVC()
        model.fit(X_train, Y_train)
    elif algorithm == "Multilayer Perceptron (MLP)":
        model = MLPClassifier(random_state=1,hidden_layer_sizes=(64,32), activation='logistic', max_iter=2000)
        model.fit(X_train, Y_train)
    elif algorithm == "Extreme Gradient Boost (XGBoost)":
        model = XGBClassifier(learning_rate=0.05,max_depth=8,min_child_weight=5)
        model.fit(X_train, Y_train)
    elif algorithm == "CatBoost":
        model = CatBoostClassifier(iterations = 1000,random_strength = 1)
        model.fit(X_train, Y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(Y_test, y_pred)
    precision = precision_score(Y_test, y_pred)
    recall = recall_score(Y_test, y_pred)
    f1 = f1_score(Y_test, y_pred)
    return y_pred,accuracy, precision, recall, f1

def plot_confusion_matrix(Y_test,y_pred):
    cm = confusion_matrix(Y_test,y_pred)
    #Plot the confusion matrix.
    st.write(f"Confusion Matrix:")
    fig = plt.figure(figsize=(2,2))
    sns.heatmap(cm,annot=True,fmt='d',
                xticklabels=['1','0'],
                yticklabels=['1','0'])
    plt.ylabel('Prediction',fontsize=6)
    plt.xlabel('Actual',fontsize=6)
    plt.title('Confusion Matrix',fontsize=9)
    st.pyplot(fig) 

def all_modelMaking():

    X_train,X_test,Y_train,Y_test = data_reading()

    model_knn = KNeighborsClassifier(n_neighbors=8)
    model_nb = GaussianNB()
    model_svc = SVC()
    model_mlp = MLPClassifier(random_state=1,hidden_layer_sizes=(64,32), activation='logistic', max_iter=2000)
    model_xg = XGBClassifier(learning_rate=0.05,max_depth=8,min_child_weight=5)
    model_cat = CatBoostClassifier(iterations = 1000,random_strength = 1)    

    model_knn.fit(X_train, Y_train)
    model_nb.fit(X_train, Y_train)
    model_svc.fit(X_train, Y_train)
    model_mlp.fit(X_train, Y_train)
    model_xg.fit(X_train, Y_train)
    model_cat.fit(X_train, Y_train)

    y_pred_knn = model_knn.predict(X_test)
    accuracy_knn = accuracy_score(Y_test, y_pred_knn)
    precision_knn = precision_score(Y_test, y_pred_knn)
    recall_knn = recall_score(Y_test, y_pred_knn)
    f1_knn = f1_score(Y_test, y_pred_knn)
    # return accuracy_knn, precision_knn, recall_knn, f1_knn
    y_pred_nb = model_nb.predict(X_test)
    accuracy_nb = accuracy_score(Y_test, y_pred_nb)
    precision_nb = precision_score(Y_test, y_pred_nb)
    recall_nb = recall_score(Y_test, y_pred_nb)
    f1_nb = f1_score(Y_test, y_pred_nb)
    # return accuracy_nb, precision_nb, recall_nb, f1_nb
    y_pred_svc = model_svc.predict(X_test)
    accuracy_svc = accuracy_score(Y_test, y_pred_svc)
    precision_svc = precision_score(Y_test, y_pred_svc)
    recall_svc = recall_score(Y_test, y_pred_svc)
    f1_svc = f1_score(Y_test, y_pred_svc)
    # return accuracy_svc, precision_svc, recall_svc, f1_svc
    y_pred_mlp = model_mlp.predict(X_test)
    accuracy_mlp = accuracy_score(Y_test, y_pred_mlp)
    precision_mlp = precision_score(Y_test, y_pred_mlp)
    recall_mlp = recall_score(Y_test, y_pred_mlp)
    f1_mlp = f1_score(Y_test, y_pred_mlp)
    # return accuracy_mlp, precision_mlp, recall_mlp, f1_mlp
    y_pred_xg = model_xg.predict(X_test)
    accuracy_xg = accuracy_score(Y_test, y_pred_xg)
    precision_xg = precision_score(Y_test, y_pred_xg)
    recall_xg = recall_score(Y_test, y_pred_xg)
    f1_xg = f1_score(Y_test, y_pred_xg)

    y_pred_cat = model_cat.predict(X_test)
    accuracy_cat = accuracy_score(Y_test, y_pred_cat)
    precision_cat = precision_score(Y_test, y_pred_cat)
    recall_cat = recall_score(Y_test, y_pred_cat)
    f1_cat = f1_score(Y_test, y_pred_cat)
    # return accuracy_cat, precision_cat, recall_cat, f1_cat
    st.header('k-NN')
    st.write(f"Accuracy of kNN: {100*accuracy_knn}")
    st.write(f"Precision of kNN: {100*precision_knn}")
    st.write(f"Recall of kNN: {100*recall_knn}")
    st.write(f"F1 Score of kNN: {100*f1_knn}")
    plot_confusion_matrix(Y_test,y_pred_knn)
    #seperator between various algorithms]
    st.markdown("""<hr style="height:10px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)

    st.header('Naive Bayes')
    st.write(f"Accuracy of Naive Bayes: {100*accuracy_nb}")
    st.write(f"Precision of Naive Bayes: {100*precision_nb}")
    st.write(f"Recall of Naive Bayes: {100*recall_nb}")
    st.write(f"F1 Score of Naive Bayes: {100*f1_nb}")
    plot_confusion_matrix(Y_test,y_pred_nb)
    st.markdown("""<hr style="height:10px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)
           
    st.header('Support Vector Machine')
    st.write(f"Accuracy of SVM: {100*accuracy_svc}")
    st.write(f"Precision of SVM: {100*precision_svc}")
    st.write(f"Recall of SVM: {100*recall_svc}")
    st.write(f"F1 Score of SVM: {100*f1_svc}")
    plot_confusion_matrix(Y_test,y_pred_svc)
    st.markdown("""<hr style="height:10px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)

    st.header('Multilayer Perceptron')
    st.write(f"Accuracy of MLP: {100*accuracy_mlp}")
    st.write(f"Precision of MLP: {100*precision_mlp}")
    st.write(f"Recall of MLP: {100*recall_mlp}")
    st.write(f"F1 Score of MLP: {100*f1_mlp}")
    plot_confusion_matrix(Y_test,y_pred_mlp)
    st.markdown("""<hr style="height:10px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)

    st.header('Extreme Gradient Boost')           
    st.write(f"Accuracy of XGB: {100*accuracy_xg}")
    st.write(f"Precision of XGB: {100*precision_xg}")
    st.write(f"Recall of XGB: {100*recall_xg}")
    st.write(f"F1 Score of XGB: {100*f1_xg}")
    plot_confusion_matrix(Y_test,y_pred_xg)
    st.markdown("""<hr style="height:10px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)

    # Display the algorithm's performance
    st.header('Catboost')
    st.write(f"Accuracy of Catboost: {100*accuracy_cat}")
    st.write(f"Precision of Catboost: {100*precision_cat}")
    st.write(f"Recall of Catboost: {100*recall_cat}")
    st.write(f"F1 Score of Catboost: {100*f1_cat}")
    plot_confusion_matrix(Y_test,y_pred_cat)

    st.header('Bar Plot Comparing all algorithms')
    fig = plt.figure(figsize=(5,5))
    plt.bar("KNN",accuracy_knn,width=0.7)
    plt.bar("NB",accuracy_nb,width=0.7)
    plt.bar("SVM",accuracy_svc,width=0.7)
    plt.bar("MLP",accuracy_mlp,width=0.7)
    plt.bar("XGB",accuracy_xg,width=0.7)
    plt.bar("Catboost",accuracy_cat,width=0.7)
    plt.xlabel("Machine Learning Algorithm")
    plt.ylabel('Accuracy')
    st.pyplot(fig)

def all_model_withoversampling():
    
    X_train_scaled,X_test_scaled,Y_train_resampled,X_test,Y_test = oversampled_data_reading()

    # Naive Bayes 
    model_nb= GaussianNB()
    model_nb.fit(X_train_scaled,Y_train_resampled)

    #KNN
    model_knn= KNeighborsClassifier(n_neighbors=8)
    model_knn.fit(X_train_scaled,Y_train_resampled)

    # Train an XGBoost model on the resampled and scaled data
    model_xg = XGBClassifier(learning_rate=0.05,max_depth=8,min_child_weight=5)
    model_xg.fit(X_train_scaled, Y_train_resampled)

    # CAT Boost
    model_cat = CatBoostClassifier(iterations = 1000,random_strength = 1)
    model_cat.fit(X_train_scaled,Y_train_resampled)

    # SVM
    model_svm=SVC()
    model_svm.fit(X_train_scaled,Y_train_resampled)

    # MLP
    model_mlp=MLPClassifier(random_state=1,hidden_layer_sizes=(64,32), activation='logistic', max_iter=2000)
    model_mlp.fit(X_train_scaled,Y_train_resampled)

    # Evaluate the model on the test set
    accuracy_knn = model_knn.score(X_test_scaled, Y_test)
    y_pred_knn = model_knn.predict(X_test)
    precision_knn = precision_score(Y_test, y_pred_knn)
    recall_knn = recall_score(Y_test, y_pred_knn)
    f1_knn = f1_score(Y_test, y_pred_knn)
            

    # Evaluate the model on the test set
    accuracy_nb = model_nb.score(X_test_scaled, Y_test)
    y_pred_nb = model_xg.predict(X_test)
    precision_nb = precision_score(Y_test, y_pred_nb)
    recall_nb = recall_score(Y_test, y_pred_nb)
    f1_nb = f1_score(Y_test, y_pred_nb)

    # Evaluate the model on the test set
    accuracy_svm = model_svm.score(X_test_scaled, Y_test)
    y_pred_svm = model_xg.predict(X_test)
    precision_svm = precision_score(Y_test, y_pred_svm)
    recall_svm = recall_score(Y_test, y_pred_svm)
    f1_svm = f1_score(Y_test, y_pred_svm)

    # Evaluate the model on the test set
    accuracy_mlp = model_mlp.score(X_test_scaled, Y_test)
    y_pred_mlp = model_mlp.predict(X_test)
    precision_mlp = precision_score(Y_test, y_pred_mlp)
    recall_mlp = recall_score(Y_test, y_pred_mlp)
    f1_mlp = f1_score(Y_test, y_pred_mlp)

    # Evaluate the model on the test set
    accuracy_xg= model_xg.score(X_test_scaled, Y_test)
    y_pred_xg = model_xg.predict(X_test)
    precision_xg = precision_score(Y_test, y_pred_xg)
    recall_xg = recall_score(Y_test, y_pred_xg)
    f1_xg = f1_score(Y_test, y_pred_xg)
            
    # Evaluate the model on the test set
    accuracy_cat = model_cat.score(X_test_scaled, Y_test)
    y_pred_cat = model_cat.predict(X_test)
    precision_cat = precision_score(Y_test, y_pred_cat)
    recall_cat = recall_score(Y_test, y_pred_cat)
    f1_cat = f1_score(Y_test, y_pred_cat)

    st.header('k-NN')
    st.write(f"Accuracy of kNN after target balancing: {100*accuracy_knn}")
    # st.write(f"Precision of kNN after target balancing: {100*precision_knn}")
    # st.write(f"Recall of kNN after target balancing: {100*recall_knn}")
    # st.write(f"F1 Score of kNN after target balancing: {100*f1_knn}")
    #confusion matrix
    plot_confusion_matrix(Y_test,y_pred_knn) 
    #seperator between various algorithms]
    st.markdown("""<hr style="height:10px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)

    st.header('Naive Bayes')
    st.write(f"Accuracy of Naive Bayes after target balancing: {100*accuracy_nb}")
    # st.write(f"Precision of Naive Bayes after target balancing: {100*precision_nb}")
    # st.write(f"Recall of Naive Bayes after target balancing: {100*recall_nb}")
    # st.write(f"F1 Score of Naive Bayes after target balancing: {100*f1_nb}")
    #confusion matrix
    plot_confusion_matrix(Y_test,y_pred_nb)
    st.markdown("""<hr style="height:10px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)
            
    st.header('Support Vector Machine')
    st.write(f"Accuracy of SVM after target balancing: {100*accuracy_svm}")
    # st.write(f"Precision of SVM after target balancing: {100*precision_svm}")
    # st.write(f"Recall of SVM after target balancing: {100*recall_svm}")
    # st.write(f"F1 Score of SVM after target balancing: {100*f1_svm}")
    #confusion matrix
    plot_confusion_matrix(Y_test,y_pred_svm)
    st.markdown("""<hr style="height:10px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)

    st.header('Multilayer Perceptron')
    st.write(f"Accuracy of MLP after target balancing: {100*accuracy_mlp}")
    # st.write(f"Precision of MLP after target balancing: {100*precision_mlp}")
    # st.write(f"Recall of MLP after target balancing: {100*recall_mlp}")
    # st.write(f"F1 Score of MLP after target balancing: {100*f1_mlp}")
    plot_confusion_matrix(Y_test,y_pred_mlp)
    st.markdown("""<hr style="height:10px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)

    st.header('Extreme Gradient Boost')           
    st.write(f"Accuracy of after target balancing: {100*accuracy_xg}")
    # st.write(f"Precision of after target balancing: {100*precision_xg}")
    # st.write(f"Recall of after target balancing: {100*recall_xg}")
    # st.write(f"F1 Score of after target balancing: {100*f1_xg}")
    plot_confusion_matrix(Y_test,y_pred_xg)
    st.markdown("""<hr style="height:10px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)

    # Display the algorithm's performance
    st.header('Catboost')
    st.write(f"Accuracy of Catboost after target balancing: {100*accuracy_cat}")
    # st.write(f"Precision of Catboost after target balancing: {100*precision_cat}")
    # st.write(f"Recall of Catboost after target balancing: {100*recall_cat}")
    # st.write(f"F1 Score of Catboost after target balancing: {100*f1_cat}")
    plot_confusion_matrix(Y_test,y_pred_cat)
    
    #bar plot
    st.header('Accuracies of various algorithms')
    fig = plt.figure(figsize=(5,5))
    plt.bar("KNN",accuracy_knn,width=0.7)
    plt.bar("NB",accuracy_nb,width=0.7)
    plt.bar("SVM",accuracy_svm,width=0.7)
    plt.bar("MLP",accuracy_mlp,width=0.7)
    plt.bar("XGB",accuracy_xg,width=0.7)
    plt.bar("Catboost",accuracy_cat,width=0.7)
    plt.xlabel("Machine Learning Algorithm")
    plt.ylabel('Accuracy')
    st.pyplot(fig)

def k_fold():
    df = pd.read_csv(uploaded_file)
    df["target"] = df["target"].apply(lambda x:0 if x==0 else 1)
    kf = KFold(n_splits=10,shuffle=True,random_state=None)
    y = df["target"]
    X = df.drop(["target"],axis=1)

    #Smote
    smote_enn = SMOTEENN(random_state=2)
    X_balanced,y_balanced = smote_enn.fit_resample(X,y)
    X = X_balanced
    y = y_balanced

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    cnt =1
    for train_index,test_index in kf.split(X,y):
        train_len = len(train_index)
        test_len = len(test_index)
        st.write(f"Fold : {cnt} , Train set : {train_len} Test_index : {test_len}")
        cnt+=1
    model1 = CatBoostClassifier(iterations=1000,eval_metric='Accuracy',verbose=0)
    scores = cross_val_score(model1,X,y,scoring='accuracy',cv=kf,n_jobs=-1)
    mean_score = (mean(scores))*100
    st.write(f"Accuracy:{mean_score}")

    st.header("K Fold plot")
    sns.set_style('whitegrid')
    sns.set_context("talk")
    sns.set(font_scale = 1.5)
    fig = plt.figure(figsize=(16,11), dpi=141)
    plt.title('k-fold cross validation')
    plt.plot(scores)
    st.pyplot(fig)

# Create a submit button
if submit_btn:
    if uploaded_file is not None:
        if algorithm =="K_Fold":
            k_fold()
        elif algorithm=="Compare All Algorithms" and targetBal == "No (Base paper implementation)":
            all_modelMaking()
        elif algorithm =="Compare All Algorithms" and targetBal == "Yes":
            all_model_withoversampling()
        elif targetBal == "Yes":
            # Load the data into a pandas DataFrame
            data = pd.read_csv(uploaded_file)
            data["target"] = data["target"].apply(lambda x: 0 if x==0 else 1)
            # Separate features and target variable
            X = data.drop('target', axis=1)
            Y = data['target']
            # Combine SMOTE and ENN to balance the dataset
            smote_enn = SMOTEENN(random_state=42)
            X_balanced, Y_balanced = smote_enn.fit_resample(X, Y)

            # Split the data into training and testing sets
            X_train, X_test, Y_train, Y_test = train_test_split(X_balanced, Y_balanced, test_size=0.2, random_state=42)

            # Perform oversampling of the minority class using SMOTE
            #smote = SMOTE(random_state=42)
            #X_train_resampled, Y_train_resampled = smote.fit_resample(X_train, Y_train)

            X_train_resampled, Y_train_resampled  = X_train,Y_train
            
            # Perform standard scaling on the features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_resampled)
            X_test_scaled = scaler.transform(X_test) 
            st.write(algorithm)
            y_pred, accuracy, precision, recall, f1 = train_test_model(algorithm, X_train_scaled, X_test_scaled, Y_train_resampled, Y_test)
            
            st.header(algorithm)
            st.write(f"Accuracy of {algorithm} after target balancing: {100*accuracy}")
            st.write(f"Precision of {algorithm} after target balancing: {100*precision}")
            st.write(f"Recall of {algorithm} after target balancing: {100*recall}")
            st.write(f"F1 Score of {algorithm} after target balancing: {100*f1}")
            plot_confusion_matrix(Y_test,y_pred)
        else:        
            data = pd.read_csv(uploaded_file)
            data["target"] = data["target"].apply(lambda x: 0 if x==0 else 1)
            # Separate features and target variable
            X = data.drop("target", axis=1)
            Y = data["target"]
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
            # Split the data into training and testing sets
            scaler = StandardScaler()
            X_train=scaler.fit_transform(X_train)
            X_test=scaler.transform(X_test)
            y_pred, accuracy, precision, recall, f1 = train_test_model(algorithm, X_train, X_test, Y_train, Y_test)
            # Display the algorithm's performance
            st.header(algorithm)
            st.write(f"Accuracy: {100*accuracy}")
            st.write(f"Precision: {100*precision}")
            st.write(f"Recall: {100*recall}")
            st.write(f"F1 Score: {100*f1}")
            plot_confusion_matrix(Y_test,y_pred)
    else:
        st.write("Please upload a CSV file.")
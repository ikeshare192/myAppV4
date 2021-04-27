from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import time
import numpy as np
import pandas as pd
import streamlit as st

#The header 
st.title("AI and Machine Learning for Oil and Gas")
st.header("An intoductory learning app for ops and engineering")
st.write(" ")

st.subheader("**ReadMe:**")
st.write("""
The purpose of this application is to provide a simplistic introduction to \
machine learning.  You will begin by uploading your  \
training data set from a local file system.  After the upload look at the \
pulldown menu and choose the variable(column) that you want to predict.  \
After selecting the variable that you want to predict, go to the left-side \
bar and choose an algorithm.  The MLP Regression model is a neural network \
model, which is typically used for more complex data sets.  The Ridge regression \
model is used for simpler data sets.  When you are ready to make predictions, just \
adjust the sliders in the left-side bar to see how different parameters impact the predicted value.
""")
st.write("""
**Note: Any data uploaded to this application is removed from memory when the browser \
is refreshed or closed**.    
""")
st.write("""
**Note: Large data sets can consume a lot of computational resources and time.  Please observe \
the swim/bike/run/row processing indicator in the upper right corner while processing.**
""")

st.subheader("**1. Select the file type below and upload the training data**")

#main dropdown menu to select the data format
choose_file_type  = st.selectbox(
    "Choose the data format",
    [" ",
    "Excel",
    "CSV"]
    )

#sidebar dropdown menu to select the regression algorith
choose_model = st.sidebar.selectbox(
    "Choose the machine learning model",
    [" ",
    "MLP Regression",
    "Ridge Regression"]
         )
def X_checkers(df):
    names = df.columns
    checks = []
    selected = []
    for name in names:
        box = name.encode("utf-8")
        cheks = st.sidebar.checkbox(box, key=box)
        checks.append(cheks)
    dict_box = dict(zip(names, checks))
    for key, value in dict_box.items():
        if value == True:
            selected.append(key)
    X = df[selected]
    #st.write("This is a preview of your training columns")
    #X
    return selected, X

def choose_target(columns):
    y = st.selectbox("", columns)
    return y
    
#creates a sidebar slider
def get_user_input(X):
    value = []
    names = X.columns
    dictionary = {}
    #iteration to create the sliders
    for name in names:
        minimum = np.min(X[name]).astype(int).item() #as item because of the int32, int mismatch
        maximum = np.max(X[name]).astype(int).item()
        sliders = st.sidebar.slider(str(name), minimum, maximum, 1)
        value.append(sliders)
    #return value
    dictionary = dict(zip(names, value))
    features = pd.DataFrame(dictionary, index=[0])
    return features

@st.cache(suppress_st_warning=True)
#define the training function for the MLP classifier
def mlp_train_metrics(X,y):
    """ Returns the training metrics for MLP Regressor """

    X_train, X_test, y_train, y_test = train_test_split(X,y, random_state = 42)
    #model pipeline
    numeric_transformer = Pipeline(
        steps = [
            ('imputer', SimpleImputer(missing_values= np.nan, strategy='mean')), 
            ('polynomial', PolynomialFeatures(degree = 2, interaction_only = False)),
            ('scaler', StandardScaler())
                ]
                                   )
    #mlp model regressor
    mlp_pipe = Pipeline(
        steps = [
            ('numeric', numeric_transformer),
            ('classifier', MLPRegressor(solver = 'adam',
                                    max_iter = 350,
                                    #learning_rate = 2e-3,
                                    hidden_layer_sizes = (90,60,30)))
                ]
                    )
    #pip fit
    mlp_fit = mlp_pipe.fit(X_train, y_train)
    #R^2 of the model
    mlp_score = mlp_pipe.score(X_test, y_test)
    return mlp_score, mlp_fit
    
@st.cache(suppress_st_warning=True)
def ridge_train_metrics(X,y):
    """ returns the training metrics for Ridge Regresssion Model """

    X_train, X_test, y_train, y_test = train_test_split(X,y, random_state = 42)
    #ridge model numeric transformation pipeline
    numeric_transformer = Pipeline(
        steps = [
        ('imputer', SimpleImputer(missing_values=np.nan, strategy='mean')),    
        ('polynomial', PolynomialFeatures(degree = 3, interaction_only = False)),
        ('scaler', StandardScaler())
            ]
                                )
    #ridge model pipeline
    ridge_pipe = Pipeline(
        steps = [
            ('numberic', numeric_transformer),
            ('classifieer', Ridge(alpha = 10,
                            solver = "cholesky",
                            random_state=42))
                ]
                        )            
    #model fit
    ridge_fit = ridge_pipe.fit(X_train, y_train)
    #ridge model evaluations
    ridge_score = ridge_pipe.score(X_test, y_test)
    return ridge_score, ridge_fit
          
def mlp_page_builder(X,y):
    #function that builds the MLP Model when MLP is selected from the dropdown menu
    st.write("*"*80)
    mlp_score, mlp_reg = mlp_train_metrics(X,y)
    st.subheader("**3. Interact with the model in the steps below**")
    st.subheader('**Multi-Layer Perceptron(MLP) Model Introduction:**')
    st.write(
        'Multi-layer Perceptron (MLP) is a supervised learning algorithm that learns a function\
        by training on a dataset, where the number of dimensions for input is the number of\
        dimensions for output. Given a set of features and a target , it can develop a non-linear\
        function approximator for either classification or regression. It is different from\
        logistic regression, in that between the input and the output layer, there can be one or\
        more non-linear layers, called hidden layers.')
    #for formatting on the main page
    st.write("-"*80)
    st.write("**Training is complete!!**")
    #training R^2
    st.write('Model R^2 = ', f'{mlp_score:.2f}')
    st.text('Note: The higher the value the higher the model accuracy')
    #for formatting on the main page
    st.write("-"*80)
    return mlp_reg

def ridge_page_builder(X,y):
    #function that builds the Ridge Regression Model when Ridge is selected from the dropdown menu
    ridge_score, ridge_reg = ridge_train_metrics(X,y)
    st.write("*"*80)
    st.subheader("**3. Interact with the model in the steps below**")
    st.subheader('**__Ridge Regression Model Introduction:__**')
    st.write(
        'This model solves a regression model where the loss function is the linear\
         least squares function and regularization is given by the l2-norm. Also known as\
              Ridge Regression or Tikhonov regularization.'
                   )
    #for formatiing
    st.write("-"*80)
    #text to tell the user training is complete
    st.write("**Training is complete!!**")               
    st.write('Model R^2 = ', f'{ridge_score:.2f}')
    st.text('Note: The higher the value the higher the model accuracy')
    #for formatting
    st.write("-"*80)
    return ridge_reg

#main body function
def main():
    st.write("-"*80)

    def upload_csv():
        st.set_option('deprecation.showfileUploaderEncoding', False)
        uploaded_file = st.file_uploader("Choose a csv file", type="csv")
        df = pd.DataFrame()
        if uploaded_file is not None:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file)
            st.text(f"You have successfully uploaded {len(df)} rows and {len(df.columns)} columns")
            st.subheader("**3. Go to the leftsidebar and select the columns to include \
for training the model.**") 
        columns = df.columns
        X = X_checkers(df)
        return df, columns, X

    def upload_xls():
        st.set_option('deprecation.showfileUploaderEncoding', False)
        uploaded_file = st.file_uploader("Choose a .xls file then select continue", type="xlsx")
        df = pd.DataFrame()
        if uploaded_file is not None:
            uploaded_file.seek(0)
            df = pd.read_excel(uploaded_file)
            st.text(f"You have successfully uploaded {len(df)} rows and {len(df.columns)} columns")
            #st.subheader("**2. Select the feature that you would like to predict \
#then go to the left-side bar and train an algorithm**") 
        columns = df.columns
        st.sidebar.text("Choose the training features")
        return df, columns

    #cont1 = st.checkbox("Click here after a successful upload")
    if choose_file_type == "CSV":
        st.subheader("**2. Upload your training data below.**")
        #upload file
        X = upload_csv()
        

    if choose_file_type == "Excel":
        st.write("** Upload your training data below.**")
        #upload file
        df,columns = upload_xls()


    # Page for MLP Regression
    if choose_model == "MLP Regression":
        mlp_model = mlp_page_builder(X,y)
        #st.write("**3. Choose to make predictions below or run metrics from the leftside bar**")
        if st.checkbox("Click here, then go the left-side bar \
            and adjust the sliders to influence the predicted values."):
            features = get_user_input(X)
            mlp_score, mlp_fit = mlp_train_metrics(X,y)
            output = mlp_fit.predict(features)
            st.write(f'The predicted **{targ_var}** value is: **{output.item():.2f}**')

    #page for Ridge Regression
    if choose_model == "Ridge Regression":
        ridge_model = ridge_page_builder(X,y)
        #st.write("**3. Choose to make predictions below or run metrics from the leftside bar**")
        if st.checkbox("Click here, then go to the left-side bar \
            and adjust the sliders to influence the predicted values."):
            features = get_user_input(X)
            ridge_score, ridge_fit = ridge_train_metrics(X,y)
            output = ridge_fit.predict(features)
            st.write(f'The predicted **{targ_var}** value is: **{output.item():.2f}**')


if __name__ == "__main__":
    main()




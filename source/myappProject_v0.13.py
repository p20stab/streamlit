import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import scanpy as sc
import time
import umap

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

st.set_page_config(
    page_title="Dataset Visualization",
    page_icon="🧇", # :waffle:
    layout="centered",
    initial_sidebar_state="expanded",
    menu_items={
        # 'Get Help': 'https://www.extremelycoolapp.com/help',
        # 'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "### Team\nΣταμπουλης Θεοδωρος Π2020062"
    }
)

# Global variables
if 'df' not in st.session_state: st.session_state['df'] = None
if 'df_cols' not in st.session_state: st.session_state['df_cols'] = [None,"aw"]
if 'isdev' not in st.session_state: st.session_state['isdev'] = False

# Sidebar menu
with st.sidebar:
    menu_selected = option_menu(
        menu_title = "Dataset Visualization",
        options = ["Dataset","Pre-Proccess","Statistics","Plots","Classification"],
        menu_icon = "menu-button-wide"
        #icons = ["cloud_upload","house"]
        # https://icons.getbootstrap.com/
    )

    with st.expander("Extra Settings"):
        st.write("*Dark/White mode and Wide/Centered screen settings are at top right*")
        st.session_state.isdev = st.toggle("Unsafe/Developer Mode")

# Page
if menu_selected == "Dataset":
    st.title("Basic Data Uploader")

    uploaded_file = st.file_uploader(label="Upload file CSV, XLSX, TXT", type=["csv","xlsx","txt"], label_visibility="hidden")
    if uploaded_file is not None:
        #st.write("### file type: ", st.session_state.uploaded_file.type )
        if uploaded_file.type == "text/csv": #csv
            st.session_state.df = pd.read_csv(uploaded_file)
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": #xlsx
            st.session_state.df = pd.read_excel(uploaded_file)
        elif uploaded_file.type == "text/plain": #txt
            txt_delimiter = st.text_input("Whats the delimiter? (default=tab)","\t")
            st.session_state.df = pd.read_csv(uploaded_file, engine="python", delimiter=txt_delimiter)
        else:
            st.write("Error: File format not accepted")

    g1, g2 = st.columns([1,6])
    with g1:
        if st.button("Delete file", type="primary"): st.session_state.df = None
    with g2:
        if st.session_state.df is not None:
            st.download_button("Save file", data=st.session_state.df.to_csv(), file_name="dataset "+str(time.ctime())+".csv")
        else:
            st.button("Save file")

    st.write("### Uploaded Data:")
    if st.session_state.df is not None:

        for i in range (len(st.session_state.df.columns)):
            colmtype = st.session_state.df.dtypes.iloc[i]
            if not st.session_state.isdev and (colmtype=="string" or colmtype=="object"):
                st.warning("Warning: String data detected in dataset. Categorial data isnt supported by sklearn. \
                **There is an auto convert categorial to int(0,1,2...) in place** \
                but know that unless its [low,mid,high] or similar in that order, results may not be good.")

        if st.toggle("Show All (default 20 rows)"):
            st.dataframe(st.session_state.df)
        else:
            st.dataframe(st.session_state.df.head(20))

        # always clears and remakes it here (its fine just some cpu wasted on a task)
        st.session_state.df_cols = [None]
        st.session_state.df_cols.extend(st.session_state.df.columns)

# Page
elif menu_selected == "Pre-Proccess":
    st.title("Pre-Proccess dataset")

    df = st.session_state.df
    df_cols = st.session_state.df_cols
    isdev = st.session_state.isdev
    if df is None and not isdev:
        st.write("Dataset not uploaded")

    else:
        g1, g2, g3 = st.columns(3)

        with g1:
            if st.button("Delete NaN", type="primary"):
                st.session_state.df = df.dropna().reset_index(drop=True)
            st.write("Number of NaN", st.session_state.df.isnull().sum())
        with g2:
            if st.button("Delete duplicates", type="primary"):
                st.session_state.df = df.drop_duplicates().reset_index(drop=True)
            st.write("Number of duplicates: "+ str(st.session_state.df.duplicated().sum()))
            st.write("Number of current entries/rows: "+ str(st.session_state.df.shape[0]))
        with g3:
            df_feat = st.selectbox("Choose Feature", df_cols)

            if st.button("Drop", type="primary") and df_feat != None:
                st.session_state.df = df.drop(columns=df_feat)
                st.session_state.df_cols = [None]
                st.session_state.df_cols.extend(st.session_state.df.columns)

            if st.button("Subsample", type="primary") and df_feat != None:

                tempdf = pd.DataFrame()
                for c in df[df_feat].unique():
                    c_df = df[df[df_feat] == c]
                    n=df[df_feat].value_counts().iloc[len(df[df_feat].unique())-1]
                    c_df = c_df.sample(n=n, random_state=1)
                    tempdf = pd.concat([tempdf,c_df]).reset_index(drop = True)
                st.session_state.df = tempdf.copy()
                st.write("Subsampled")

            if df_feat != None:
                st.write (df[df_feat].value_counts().sort_index())

# Page
elif menu_selected == "Statistics":
    st.title("Statistics of features")
    
    df = st.session_state.df
    df_cols = st.session_state.df_cols
    isdev = st.session_state.isdev
    if df is None and not isdev:
        st.write("Dataset not uploaded")

    else:
        choice_stat = st.pills("Choose statistic", ["Describe","Correletion"], default="Describe", label_visibility="hidden")

        dftemp = df.copy()
        for i in range (len(dftemp.columns)):
            colm = dftemp.columns[i]
            colmtype = dftemp.dtypes.iloc[i]
            #st.write(colmtype) # do the warning at page Dataset
            if colmtype=="string" or colmtype=="object":
                count = 0
                for i in pd.unique(df[colm]):
                    dftemp[colm] = dftemp[colm].replace({i:str(count)})
                    count+=1
                dftemp[colm] = dftemp[colm].astype("int")
        df = dftemp.copy()

        if choice_stat == "Describe":
            if st.toggle("Vertical columns"):
                st.write(df.describe().transpose())
            else:
                st.write(df.describe())

        elif choice_stat == "Correletion":

            g1, g2 = st.columns(2)
            with g1: fig_height = st.slider("height", min_value=1, max_value=2*len(df.columns), value=len(df.columns))
            with g2: fig_width = st.slider("width", min_value=1, max_value=2*len(df.columns), value=len(df.columns)+len(df.columns)//2)

            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            sns.heatmap(df.corr(), annot=True, ax=ax, cmap='RdBu_r', vmin=-1.0, vmax=1.0, center=0.0)
            st.pyplot(fig)

# Page
elif menu_selected == "Plots":
    st.title("Plots")

    df = st.session_state.df
    df_cols = st.session_state.df_cols
    isdev = st.session_state.isdev
    if df is None and not isdev:
        st.write("Dataset not uploaded")

    else:
        choice_plot = st.pills("Choose plot type", ["Bars","Scatter","PairPlot","UMAP"], default="Bars", label_visibility="hidden")
        
        if choice_plot == "Bars":
            g1, g2, g3 = st.columns(3)
            with g1: c_axis = st.selectbox("Color-axis", df_cols)
            with g2: x_axis = st.selectbox("X-axis", df_cols)
            with g3: y_axis = st.selectbox("Y-axis", df_cols, index=1)
            st.bar_chart(df, x=x_axis, y=y_axis, color=c_axis)

        elif choice_plot == "Scatter":
            choice_dim = st.radio("Choose dimensions", ["2d","3d"], horizontal=True)
            
            g1, g2, g3, g4, g5 = st.columns(5)
            with g1: c_axis = st.selectbox("Color-axis", df_cols)
            with g2: x_axis = st.selectbox("X-axis", df_cols)
            with g3: y_axis = st.selectbox("Y-axis", df_cols, index=1)

            if choice_dim == "2d":
                with g4: z_axis = st.selectbox("Z-axis", df_cols, disabled=True)
                with g5: s_axis = st.selectbox("Symbol-axis", df_cols, disabled=True)
                fig = px.scatter(df, x=x_axis, y=y_axis, color=c_axis)
                st.plotly_chart(fig)

            elif choice_dim =="3d":
                with g4: z_axis = st.selectbox("Z-axis", df_cols)
                with g5: s_axis = st.selectbox("Symbol-axis", df_cols)
                fig = px.scatter_3d(df, x=x_axis, y=y_axis, z=z_axis, color=c_axis, symbol=s_axis)
                st.plotly_chart(fig)

        elif choice_plot == "PairPlot":
            c_axis = st.selectbox("Color-axis", df_cols)
            fig = sns.pairplot(df, hue=c_axis)
            st.pyplot(fig)

        elif choice_plot == "UMAP":

            # Turn strings to ints
            dftemp = df.copy()
            for i in range (len(dftemp.columns)):
                colm = dftemp.columns[i]
                colmtype = dftemp.dtypes.iloc[i]
                if colmtype=="string" or colmtype=="object":
                    count = 0
                    for i in pd.unique(df[colm]):
                        dftemp[colm] = dftemp[colm].replace({i:str(count)})
                        count+=1
                    dftemp[colm] = dftemp[colm].astype("int")
            df = dftemp.copy()

            c_axis = st.selectbox("Color-axis", df_cols)
            if c_axis is not None:

                g1, g2 = st.columns([9,1])
                with g1:
                    #sc.pl.umap(adata,color=['batch'],legend_fontsize=10) #failed... might look later
                    reducer = umap.UMAP(random_state=42)

                    scaled_tempdf = StandardScaler().fit_transform(df[df_cols[1:]].values)
                    embedding = reducer.fit_transform(scaled_tempdf)
                    
                    fig, axes = plt.subplots(1, 1)
                    plt.scatter(
                        embedding[:, 0],
                        embedding[:, 1],
                        c=[sns.color_palette()[x] for x in df[c_axis]])
                    plt.gca().set_aspect('equal', 'datalim')
                    st.pyplot(fig)

                with g2:
                    st.write("") #empty line for better look
                    colors = sns.color_palette()[0:len(df[c_axis].unique())]
                    hex_colors = [f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}' for r, g, b in colors]
                    for label, color in zip(st.session_state.df[c_axis].unique(), colors):
                        st.markdown(f"<span style='color:rgb({color[0]*255},{color[1]*255},{color[2]*255})'>{label}</span>", unsafe_allow_html=True)


# Page
elif menu_selected == "Classification":
    st.title("Machine learning - Classification")

    df = st.session_state.df
    df_cols = st.session_state.df_cols
    isdev = st.session_state.isdev
    if df is None and not isdev:
        st.write("Dataset not uploaded")

    else:
        # Turn strings to ints
        dftemp = df.copy()
        for i in range (len(dftemp.columns)):
            colm = dftemp.columns[i]
            colmtype = dftemp.dtypes.iloc[i]
            if colmtype=="string" or colmtype=="object":
                count = 0
                for i in pd.unique(df[colm]):
                    dftemp[colm] = dftemp[colm].replace({i:str(count)})
                    count+=1
                dftemp[colm] = dftemp[colm].astype("int")
        df = dftemp.copy()

        choice_clf = st.pills("Choose classifier", ["DecisionTree","RandomForest","K-NearestNeighbors","XGBoost"], default="DecisionTree", label_visibility="hidden")

        g1, g2 = st.columns(2)
        with g1: n = st.slider("k folds cross-validation", min_value=2, max_value=20, value=10)
        with g2: df_class = st.selectbox("Select Class", df_cols)
        st.write()
        if df_class is not None and choice_clf is not None:
            
            Cdepth = None
            Cneighbors = None
            if choice_clf in ["DecisionTree","RandomForest","XGBoost"]: Cdepth = st.slider("Max depth", min_value=1, max_value=len(df_cols)*2-2, value=len(df_cols)-2)
            elif choice_clf in ["K-NearestNeighbors"]: Cneighbors = st.slider("N neighbors", min_value=1, max_value=len(df[df_class].values)//5, value=len(df[df_class].values)//25)
            dict_clf = {
                "DecisionTree": DecisionTreeClassifier(max_depth=Cdepth),
                "RandomForest": RandomForestClassifier(max_depth=Cdepth),
                "K-NearestNeighbors": KNeighborsClassifier(n_neighbors=Cneighbors),
                "XGBoost": XGBClassifier(max_depth=Cdepth)
            }

            X = df.drop(columns=df_class)
            y = df[df_class]
            df_labels = st.session_state.df[df_class].unique()
            k_folds = KFold(n_splits = n, shuffle=True, random_state=42)

            fig, axes = plt.subplots(4, 5)
            fig.set_figheight(14)
            fig.set_figwidth(20)
            fig.tight_layout(pad=4.0)
            axes = axes.flatten()
            i=0
            mean_clf=0
            for train_index, test_index in k_folds.split(X):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                clf = dict_clf[choice_clf]
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                acc_clf = accuracy_score(y_test, y_pred)*100
                
                sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='g', cmap='Blues', ax=axes[i], xticklabels=df_labels, yticklabels=df_labels)
                axes[i].set_xlabel("Predicted label")
                axes[i].set_ylabel("True label")
                axes[i].set_title("Fold: "+str(i)+" | Acc: %.3f"%(acc_clf)+"%")
                i+=1
                mean_clf += acc_clf

            for i in range(n, 20):
                fig.delaxes(axes[i])
            st.pyplot(fig)

            st.write("Mean Accuracy: %.3f"%(mean_clf/n)+"%")

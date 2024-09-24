import pandas as pd
import os
import seaborn as sns
from pathlib import Path
from os.path import join
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from scipy.signal import savgol_filter
from scipy import sparse
from scipy.sparse.linalg import spsolve

# general commands which facilitate the script
class Helpers:
    
    def read_file(path: str, skip_header=0, dec='.', sepi=','):
        df = pd.read_csv(path,  sep=sepi, decimal=dec, skiprows=skip_header)
        return df
    
    def get_files(path: str):
        files = []
        for r, d, f in os.walk(path):
            for file in f:
                files.append(os.path.join(r, file))
        return files
    
    def save_html(html_object, path: str, name: str):
        Path(path).mkdir(parents=True, exist_ok=True)
        path = path + '\\' + name + '.html'
        print(path)
        html_object.write_html(path)
    
    def mkdir_ifnotexist(path):
        # mkdir if not exists
        Path(path).mkdir(parents=True, exist_ok=True)
        return path
    
    def save_df(df, path, name, index=True):
        Path(path).mkdir(parents=True, exist_ok=True)
        path = join(path, f'{name}.csv')
        print(path)
        df.to_csv(path, sep=';', decimal=',', index=index)

    # Smoothing the spectrum using Savitzky-Golay filter
    def smooth_spectrum(y, window_length=11, polyorder=3):
        return savgol_filter(y, window_length, polyorder)

    # Baseline correction using asymmetric least squares smoothing
    def baseline_correction(y, lam=1e6, p=0.001, niter=10):
        L = len(y)
        D = sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
        D = lam * D.dot(D.transpose()) 
        w = np.ones(L)
        W = sparse.spdiags(w, 0, L, L)
        for i in range(niter):
            W.setdiag(w) 
            Z = W + D
            z = spsolve(Z, w*y)
            w = p * (y > z) + (1-p) * (y < z)
        return z

    # Normalization of the spectrum to total area
    def area_normalization(x, y):
        cumulative_area = []
        normalized_area = []
        for i in range(len(x)):
                if i == 0:
                    area = 0
                    cumulative_area.append(area) # Sets area to 0 for the first data point
                else:
                    area = ((x[i] - x[i-1]) * (y[i] + y[i-1]) / 2)
                    cumulative_area.append(area)
        
        total_area = sum(cumulative_area)
        normalized_area = (cumulative_area/total_area)*100
        return normalized_area


# Define the path of the datafiles

path_train = 'PATH WHERE TRAIN DATA CAN BE FOUND'
path_test = 'PATH WHERE TEST DATA CAN BE FOUND'
save_path_train = 'PATH WHERE RESULTS SHOULD BE SAVED'

# create a folder with time and date
eval_ts = datetime.now().strftime("_%m-%d-%Y_%H-%M-%S")
os.environ["ROOT_PATH"] = Helpers.mkdir_ifnotexist(
    join(save_path_train, 'result' + eval_ts))

# define number of LDs and PCs
components_LDA = 1
components_PCA = 3

#read files in path, perform peprocessinf for each file and merge all files in a train and test dataset
def read_files(path: str, tag: str):
    files = Helpers.get_files(path)
    files = [f for f in files if f.find('.csv') >= 0]
    merged_df = pd.DataFrame()
    info = []

    for file in files:
        df = Helpers.read_file(file, dec='.', sepi=',')[['RT(milliseconds)', 'TIC']]
        x = df['RT(milliseconds)']
        y = df['TIC']
        df.set_index('RT(milliseconds)', inplace=True)
        new_index = np.arange(35700,1321860, 1)
        df = df.reindex(new_index)
        df = df.interpolate(method='linear',limit_direction='forward', axis=0)
        
        df.drop(df.index[1080000:], inplace=True)
        df.drop(df.index[:144300], inplace=True)
        
        y = Helpers.smooth_spectrum(y)
        baseline = Helpers.baseline_correction(y)
        y_corrected = y- baseline
        y = Helpers.area_normalization(x,y_corrected)
        baseline_y_area = Helpers.baseline_correction(y)*0.05

        
        merged_df = pd.merge(
            merged_df, df, how='outer', left_index=True, right_index=True)

        merged_df = merged_df.fillna(0)
        merged_df = merged_df.rename(
            columns={'TIC': file.split('\\')[5]})

        info.append(
            {'Class': file.split('\\')[5], 'filename': os.path.basename(file)})

    df_info = pd.DataFrame(info)
    
    def replace_below_threshold(column):
        max_val = column.max()
        threshold = 0.03 * max_val
        return column.where(column >= threshold, 0)
    
    merged_df = merged_df.apply(replace_below_threshold)
    
    Helpers.save_df(merged_df, join(
        os.environ["ROOT_PATH"], 'data'), f'extracted_features_{tag}')
    Helpers.save_df(df_info, join(
        os.environ["ROOT_PATH"], 'data'), f'extracted_features_info_{tag}')
    

# Perform Data Reduction with PCA on your Train DataFrame
def create_pca(path_merged_data_train: str, path_merged_data_train_info: str):
    df = pd.read_csv(path_merged_data_train, decimal=',', sep=';')
    df_info = pd.read_csv(path_merged_data_train_info, decimal=',', sep=';')
    df.set_index('RT(milliseconds)', inplace=True)
    pca = PCA(n_components=components_PCA).fit(df.T)
    principalComponents = pca.transform(df.T)
    df_PCA = pd.DataFrame(data=principalComponents, columns=[
        f'PC{i+1}' for i in range(components_PCA)])
    df_PCA.set_index(df_info['Class'], inplace=True)
    df_PCA.to_csv('PC.csv', index=False)

    # get variance ratio for each PC as well as the total variance
    variance_ratio = pca.explained_variance_ratio_
    for i, ratio in enumerate(variance_ratio):
        print(f"PC{i + 1}: {ratio * 100:.2f}%")     
    total_variance = np.sum(variance_ratio)   
    print(f"Total variance explained: {total_variance * 100:.2f}%")
    
    # save all PC-loadings
    pca_loadings = pca.components_
    loadings_df = pd.DataFrame(data=pca_loadings.T, columns=[f'PC{i+1}' for i in range(len(pca_loadings))])
    loadings_df.to_csv('Loadings.csv', index=False)
    return pca, df_PCA, df_info

# Perform classification with LDA on your reduced Train DataFrame (PCA DataFrame)
def create_lda(df_pca: pd.DataFrame, df_info: pd.DataFrame):
    X = df_pca.values
    y = df_pca.index
    name = df_info['filename']
    lda = LDA(n_components=components_LDA).fit(X, y)
    X_lda = lda.fit_transform(X, y)
    dfLDA_train = pd.DataFrame(data=X_lda, columns=[
        f'LD{i+1}' for i in range(components_LDA)])
    dfLDA_train['file'] = name
    dfLDA_train.index = y

    # Get your most influential PCs
    lda_loadings = lda.coef_
    most_influential_pcs = sorted(range(components_PCA), key=lambda x: abs(lda_loadings[0][x]), reverse=True)
    print(most_influential_pcs)
    
    # Perform "leave one out" crossvalidation on your LDA (Get confusion matrix)  
    y_pred = lda.predict(X)
    cm = confusion_matrix(y, y_pred)
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    print(cm)
    top_margin =0.06
    bottom_margin = 0.06
    
    # Plot the confusion matrix as a heatmap
    fig, ax = plt.subplots(
        figsize=(10,8), 
        gridspec_kw=dict(top=1-top_margin, bottom=bottom_margin))
    sns.heatmap(cm / cm_sum.astype(float), annot=True , cbar=False, cmap='gist_earth', fmt='.2%')
    ax.set_xlabel('Predicted', fontsize=15)
    ax.set_ylabel('Actual', fontsize=15)
    ax.set_title('Confusion Matrix', fontsize=15)
    cbar = ax.figure.colorbar(ax.collections[0])
    cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    cbar.set_ticklabels(["0%",'20%', '40%', '60%', '80%', "100%"], fontsize=15)
    ax.xaxis.set_ticklabels(df_pca.index.unique().tolist(), fontsize=15)
    ax.yaxis.set_ticklabels(df_pca.index.unique().tolist(), fontsize=15)
    plt.show()
    
    # perform the k-fold crossvalidation
    cv_scores = cross_val_score(lda, X, y, cv=10)
    print("Average cross-validation score:", cv_scores.mean())
    
    return lda, dfLDA_train

# Perform Data Reduction for your Test DataFrame with the PCA of the Train DataFrame
def push_to_pca(pca: PCA, path_merged_data_test: str, path_merged_data_test_info: str):
    df = pd.read_csv(path_merged_data_test, decimal=',', sep=';')
    df_info = pd.read_csv(path_merged_data_test_info, decimal=',', sep=';')
    df.set_index('RT(milliseconds)', inplace=True)
    transformed_data = pca.transform(df.T)
    dfPCA_test = pd.DataFrame(data=transformed_data, columns=[
        f'PC{i+1}' for i in range(components_PCA)], index=df_info['Class'])
 
    return dfPCA_test, df_info

# Perform classification for your reduced test DataFrame with the LDA of the train DataFrame
def push_to_lda(lda: LDA, transformed_data: pd.DataFrame, transformed_data_info: pd.DataFrame):
    predictions = lda.predict(transformed_data.values)
    transformed_data_lda = lda.transform(transformed_data.values)
    name = transformed_data_info['filename']
    df_lda_test_transformed = pd.DataFrame(data=transformed_data_lda, columns=[
        f'LD{i+1}' for i in range(components_LDA)])
    df_lda_test_transformed['file'] = name
    df_lda_test_transformed.index = transformed_data.index
    
    return df_lda_test_transformed, predictions

# Combine the Train and Test DataFrames
def combine_data(df_test: pd.DataFrame, df_train: pd.DataFrame):
    df_test['Dataset'] = ['test' for _ in df_test.index]
    df_train['Dataset'] = ['train' for _ in df_train.index]
    df_merged = pd.concat([df_train, df_test], ignore_index=False, sort=False)
    return df_merged

# Plot the LDA and save the Plot
def plot(df: pd.DataFrame):
    fig = px.scatter(df, x='LD1', color=df.index, hover_name='file', symbol='Dataset', symbol_sequence= ['circle', 'diamond'])
    fig.update_traces(marker=dict(size=12,
                              line=dict(width=1,
                                        color='DarkSlateGrey')),
                  selector=dict(mode='markers'))
    Helpers.save_html(fig, join(os.environ["ROOT_PATH"], 'plots'), 'LDA')
    fig.show()
    


# Main Program to use all definitions
if __name__ == '__main__':
    
    # produce train DataFrame
    df_raw_train = read_files(path_train, 'train')
    
    # produce test DataFrame
    df_raw_test = read_files(path_test, 'test')

    # create PCA with Train DataFrame
    pca, df_pca, df_info = create_pca(join(
        os.environ["ROOT_PATH"], 'data', f'extracted_features_train.csv'),
        join(
        os.environ["ROOT_PATH"], 'data', f'extracted_features_info_train.csv'))
    
    # create LDA with Train DataFrame
    lda, df_lda_train = create_lda(df_pca, df_info)
    
    # perform PCA with Test DataFrame
    transformded_data_test, df_info_test = push_to_pca(pca, join(
        os.environ["ROOT_PATH"], 'data', f'extracted_features_test.csv'),
        join(
        os.environ["ROOT_PATH"], 'data', f'extracted_features_info_test.csv'))
    
    # perform LDA with test DataFrame
    df_lda_test, predictions = push_to_lda(lda, transformded_data_test, df_info_test)
    
    # Combine both LDA DataFrames
    merged_df = combine_data(df_lda_test, df_lda_train)
    merged_df.to_csv('LDA_data.csv', index=False)
    
    # Plot LDA
    plot(merged_df)
    
    
    accuracy_per_class = {}
    # Get the unique classes in the test dataset
    unique_classes = set(df_lda_test.index)  

    # Calculate accuracy for each class
    for cls in unique_classes:
        indices = df_lda_test.index == cls  
        accuracy = accuracy_score(df_lda_test.index[indices], predictions[indices])  
        accuracy_per_class[cls] = accuracy

    # Print accuracy for each class
    for cls, accuracy in accuracy_per_class.items():
        print(f"Accuracy for class {cls}: {accuracy:.2f}")

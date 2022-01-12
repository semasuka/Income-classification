import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import joblib
import pickle



train_original = pd.read_csv('datasets/train.csv')

test_original = pd.read_csv('datasets/test.csv')

full_data = pd.concat([train_original, test_original], axis=0)

full_data = full_data.sample(frac=1).reset_index(drop=True)


def data_split(df, test_size):
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


train_original, test_original = data_split(full_data, 0.2)

train_copy = train_original.copy()
test_copy = test_original.copy()

gdp_data = pd.read_csv('datasets/GDP.csv')

gdp_data.sort_values(by='1990' , inplace=True,ascending=False)

gdp_data.reset_index(inplace=True, drop=True)

gdp_data.rename(columns={'Country Name':'native-country','1990':'GDP_1990'},inplace=True)


def value_cnt_norm_cal(df,feature):
    ftr_value_cnt = df[feature].value_counts()
    ftr_value_cnt_norm = df[feature].value_counts(normalize=True) * 100
    ftr_value_cnt_concat = pd.concat([ftr_value_cnt, ftr_value_cnt_norm], axis=1)
    ftr_value_cnt_concat.columns = ['Count', 'Frequency (%)']
    return ftr_value_cnt_concat

def add_gdp_data(train_copy,test_copy,gdp_data):
        full_data_copy = pd.concat([train_copy,test_copy],ignore_index=True)
        gdp_group = []
        for idx in gdp_data.index:
            if idx <= 65:
                gdp_group.append('High GDP')
            elif idx >= 65 and idx <= 130:
                gdp_group.append('Medium GDP')
            else:
                gdp_group.append('Low GDP')

        # concatenate the gdp_data with the gdp_group list
        gdp_data = pd.concat([gdp_data.rename(columns={'country':'native-country'}), pd.Series(gdp_group, name='GDP Group')], axis=1)
        # we no longer need the GDP column, so let's drop it
        gdp_data.drop(['GDP_1990'],axis=1,inplace=True)

        # we need to merge the gdp_data with X dataframe
        full_data_copy = pd.merge(full_data_copy, gdp_data, on='native-country', how='left')
        # make income_>50K the last column
        new_col_order = [col for col in full_data_copy.columns if col != 'income_>50K'] + ['income_>50K']
        return full_data_copy[new_col_order]


full_data_copy = add_gdp_data(train_copy,test_copy,gdp_data)

train_copy, test_copy = data_split(full_data_copy,0.2)

class OutlierHandler(BaseEstimator, TransformerMixin):
    def __init__(self,col_with_outliers = ['age']):
        self.col_with_outliers = col_with_outliers
    def fit(self,X,y=None):
        return self
    def transform(self,X,y=None):
        if (set(self.col_with_outliers).issubset(X.columns)):
            Q1 = X[self.col_with_outliers].quantile(.25)
            Q3 = X[self.col_with_outliers].quantile(.75)
            IQR = Q3 - Q1
            outlier_condition = (X[self.col_with_outliers] < (Q1 - 1.5 * IQR)) | (X[self.col_with_outliers] > (Q3 + 1.5 * IQR))
            index_to_keep = X[~outlier_condition.any(axis=1)].index
            return X.loc[index_to_keep]
        else:
            print("Columns not found")
            return X

class MissingValHandler(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self,X,y=None):
        return self
    def transform(self,X,y=None):
        # drop all the rows with missing values in X
        X.dropna(inplace=True)
        X.reset_index(inplace=True, drop=True)
        return X
# Input the data from streamlit interface and return the GDP group
def get_gdp_group(country_name):
    # To be implemented
    pass

class SkewnessHandler(BaseEstimator, TransformerMixin):
    def __init__(self,col_with_skewness=['age','capital-gain','capital-loss']):
        self.col_with_skewness = col_with_skewness
    def fit(self,X,y=None):
        return self
    def transform(self,X,y=None):
        if (set(self.col_with_skewness).issubset(X.columns)):
            # Handle skewness with cubic root transformation
            X[self.col_with_skewness] = np.cbrt(X[self.col_with_skewness])
            return X
        else:
            print('One or more skewed columns are not found')
            return X
class OversampleSMOTE(BaseEstimator,TransformerMixin):
    def __init__(self, perform_oversampling = True):
        self.perform_oversampling = perform_oversampling
    def fit(self,X,y=None):
        return self
    def transform(self,X,y=None):
        # function to oversample the minority class
        if self.perform_oversampling:
            smote = SMOTE()
            X_bal, y_bal = smote.fit_resample(X.iloc[:,:-1],X.iloc[:,-1])
            X_y_bal = pd.concat([pd.DataFrame(X_bal),pd.DataFrame(y_bal)],axis=1)
            return X_y_bal
        else:
            print("No oversampling performed")
            return X


def smote_pipeline_fuc(df):
    smote_pipeline = Pipeline([
        ('smote', OversampleSMOTE()) # default: perform_oversampling = True
    ])
    smote_pip_result = smote_pipeline.fit_transform(df)
    return smote_pip_result


def concat_fuc(df_ordinal_minmax, df_onehot, df_target):
    concat_df = pd.concat([df_ordinal_minmax, df_onehot, df_target], axis=1)
    return concat_df


def one_hot_enc_fuc(df):
    columns_to_one_hot_enc = ['race', 'gender', 'workclass', 'occupation','marital-status', 'relationship']
    one_hot_enc = OneHotEncoder()
    one_hot_enc.fit(df[columns_to_one_hot_enc])
    # get the result of the one hot encoding columns names
    cols_names_one_hot_enc = one_hot_enc.get_feature_names_out(columns_to_one_hot_enc)
    # change the array of the one hot encoding to a dataframe with the column names
    one_hot_result_with_names_col = pd.DataFrame(one_hot_enc.transform(df[columns_to_one_hot_enc]).toarray(),columns=cols_names_one_hot_enc)
    return one_hot_result_with_names_col


def ordinal_minmax_scaler_fuc(df):
    columns_to_ordinal_enc = ['education', 'GDP Group']
    columns_to_scale = ['age', 'capital-gain', 'capital-loss', 'hours-per-week']
    col_transformer = ColumnTransformer([
    ('Ordinal encoder',OrdinalEncoder(),columns_to_ordinal_enc), # ordinal encoding for education and GDP Group because they are ranked
    ('Min max scaler',MinMaxScaler(),columns_to_scale)]) # scaling for age, capital-gain, capital-loss, hours-per-week
    ordinal_minmax_scaler_result = col_transformer.fit_transform(df)
    ordinal_minmax_scaler_result_with_names_col = pd.DataFrame(ordinal_minmax_scaler_result,columns=columns_to_ordinal_enc+columns_to_scale)
    return ordinal_minmax_scaler_result_with_names_col


def extract_target_col(df):
    target = df.iloc[:,-1].to_frame().reset_index(drop=True)
    return target


def initial_pipeline_fuc(df):
    init_pipeline = Pipeline([
        ('Missing values handler', MissingValHandler()), # drop missing values in the whole dataset
        ('Outliers handler', OutlierHandler()),
        ('Skewness handler', SkewnessHandler()), # columns with skewness are 'age','capital-gain','capital-loss'
    ])
    init_pip_result = init_pipeline.fit_transform(df)
    return init_pip_result


def full_pipeline_fuc(df):
    # initial pipeline
    init_pip_result = initial_pipeline_fuc(df)
    #extracting the target variable
    target = extract_target_col(init_pip_result)
    # column transformers to apply ordinal and minmax transformation on specific columns
    ordinal_minmax_result = ordinal_minmax_scaler_fuc(init_pip_result)
    #one hot encoding
    one_hot_enc_result = one_hot_enc_fuc(init_pip_result)
    # concat the result from the ordinal and minmax transformation and one hot encoding with the target variable
    encoded_concat_result = concat_fuc(ordinal_minmax_result,one_hot_enc_result,target)
    # balance the imbalance data with smote function
    smote_pip_result = smote_pipeline_fuc(encoded_concat_result)
    return smote_pip_result

train_copy_prep = full_pipeline_fuc(train_copy)

test_copy_prep = full_pipeline_fuc(test_copy)

X_train_copy_prep = train_copy_prep.iloc[:,:-1]

y_train_copy_prep = train_copy_prep.iloc[:,-1]


X_test_copy_prep = test_copy_prep.iloc[:,:-1]


y_test_copy_prep = test_copy_prep.iloc[:,-1]


# %% [markdown]
# # 5. Shorlist promising models

# %% [markdown]
# ## 5.0 Functions to evaluate the models and all the metrics

# # %%
# def create_summary_table(summary_data):
#     summary_table_col = [
#         'Model name',
#         'Precision score (Validation set)',
#         'Recall score (Validation set)',
#         'F1 score (Validation set)',
#         'Accuracy score (Validation set)',
#         'AUC score (Validation set)',
#         'AUC score (Train set)',
#         'Has overfit (AUC score (Train set) > AUC score (Validation set))',
#         'Confusion matrix',
#         'Roc auc curve',
#         'Top 10 important features',
#         'Top 10 useless features',
#         'Top 10 important features plot',
#         'Top 10 useless features plot',
#         ]
#     print('\n       ***************  Metrics Summary Table  ***************\n')
#     # print all the models summary and gave the width and height to the plot
#     summary_df = pd.DataFrame(summary_data, columns=summary_table_col).iloc[[4,6,7,9,10]].style.set_properties(subset=['Confusion matrix','Roc auc curve', 'Top 10 important features plot', 'Top 10 useless features plot'], **{'width': '600px','height': '600px'})
#     # print only knn, random forest, NN, bagging, gradient boosting
#     display(HTML(summary_df.to_html()))
#     #return summary_df

# # %%
# def top_and_worst_feat_fuc(col_with_coef):
#     top_10_feat, worst_10_feat = col_with_coef[:10], col_with_coef[-10:]
#     top_10_feat_str = ""
#     worst_10_feat_str = ""
#     for count,feat in enumerate(top_10_feat, start=1):
#         # top 10 features string formatting
#         top_10_feat_str += "{0}. feature name: {1}".format(count,feat[0])+ "<br>" + "coefficient: {:.4f}".format(feat[1]) + "<br>"
#     for count,feat in enumerate(worst_10_feat, start=1):
#         # worst 10 features string formatting
#         worst_10_feat_str += "{0}. feature name: {1}".format(count,feat[0])+ "<br>" + "coefficient: {:.4f}".format(feat[1]) + "<br>"
#     return top_10_feat_str, worst_10_feat_str

# # %%
# def check_overfit(auc_score_val_set, auc_score_train_set):
#     # if the auc score of the training set is higher than the validation set by more than 0.03, then the model is overfitting
#     if (auc_score_train_set - auc_score_val_set) > 0.03:
#         return True
#     else:
#         return False

# # %%
# summary_data = []
# def growing_summary_table_fuc(model_name,precision_score,recall_score,f1_score, accuracy_score, auc_score_val_set, auc_score_train_set, is_overfitting, img_conf_matrix, img_roc_auc, col_with_coef, col_with_coef_df_top_10_html, col_with_coef_df_bottom_10_html):
#     if col_with_coef == 'No coeficient or feature importance for this model':
#         each_clf_data = [model_name,precision_score,recall_score,f1_score, accuracy_score, auc_score_val_set, auc_score_train_set, is_overfitting, img_conf_matrix, img_roc_auc, col_with_coef, col_with_coef, col_with_coef, col_with_coef]
#         summary_data.append(each_clf_data)
#     else:
#         top_10_feat, worst_10_feat = top_and_worst_feat_fuc(col_with_coef)
#         each_clf_data = [model_name,precision_score,recall_score,f1_score, accuracy_score, auc_score_val_set, auc_score_train_set, is_overfitting, img_conf_matrix, img_roc_auc, top_10_feat, worst_10_feat, col_with_coef_df_top_10_html, col_with_coef_df_bottom_10_html]
#         summary_data.append(each_clf_data)

# # %%
# def roc_curve_fuc(model_trn,model_name,X_train_copy_prep,y_train_copy_prep):
#     # path to save the roc curve
#     roc_curve_path = Path('saved_preliminary_models/{0}/{0}_roc_curve.jpg'.format(model_name))
#     try:
#         roc_curve_path.resolve(strict=True)
#     except FileNotFoundError:
#         print('\n                ROC curve\n')
#         lower_than_50k_probs = [0 for _ in range(len(y_train_copy_prep))]
#         higher_than_50k_probs = model_trn.predict_proba(X_train_copy_prep)
#         higher_than_50k_probs_pos_outcome = higher_than_50k_probs[:,1]
#         lower_than_50k_auc = roc_auc_score(y_train_copy_prep,lower_than_50k_probs)
#         higher_than_50k_probs_auc = roc_auc_score(y_train_copy_prep,higher_than_50k_probs_pos_outcome)
#         #save the auc
#         with open('saved_preliminary_models/{0}/lower_than_50k_auc_{0}.pickle'.format(model_name),'wb') as handle:
#             pickle.dump(lower_than_50k_auc,handle)
#         with open('saved_preliminary_models/{0}/higher_than_50k_probs_auc_{0}.pickle'.format(model_name),'wb') as handle:
#             pickle.dump(higher_than_50k_probs_auc,handle)
#         # print the auc
#         print('Income lower than 50k: ROC AUC=%.3f' % (lower_than_50k_auc))
#         print('Income higher than 50k: ROC AUC=%.3f' % (higher_than_50k_probs_auc))
#         lower_than_50k_false_pos_rate, lower_than_50k_true_pos_rate, _ = roc_curve(y_train_copy_prep,lower_than_50k_probs)
#         higher_than_50k_false_pos_rate, higher_than_50k_true_pos_rate, _ = roc_curve(y_train_copy_prep,higher_than_50k_probs_pos_outcome)
#         plt.plot(lower_than_50k_false_pos_rate, lower_than_50k_true_pos_rate, linestyle='--', label='Income lower than 50k')
#         plt.plot(higher_than_50k_false_pos_rate, higher_than_50k_true_pos_rate, marker='.', label='Income higher than 50k')
#         # axis labels
#         plt.xlabel('False Positive Rate (Precision)')
#         plt.ylabel('True Positive Rate (Recall)')
#         plt.title('ROC curve')
#         # show the legend
#         plt.legend()
#         # save the plot
#         plt.savefig('saved_preliminary_models/{0}/{0}_ROC_curve.jpg'.format(model_name))
#         # show the plot
#         plt.show()
#         # creating the html roc auc curve image
#         img_roc_auc = '<img src="'+ 'saved_preliminary_models/{0}/{0}_ROC_curve.jpg'.format(model_name) + '">'
#         return higher_than_50k_probs_auc, img_roc_auc
#     else:
#         # if roc curve path exists, load the auc first
#         with open('saved_preliminary_models/{0}/lower_than_50k_auc_{0}.pickle'.format(model_name),'rb') as handle:
#                 lower_than_50k_auc = pickle.load(handle)
#         with open('saved_preliminary_models/{0}/higher_than_50k_probs_auc_{0}.pickle'.format(model_name),'rb') as handle:
#                 higher_than_50k_probs_auc = pickle.load(handle)
#         # print the auc
#         print('Income lower than 50k: ROC AUC=%.3f' % (lower_than_50k_auc))
#         print('Income higher than 50k: ROC AUC=%.3f' % (higher_than_50k_probs_auc))
#         # read the ROC image
#         img_roc = mpimg.imread('saved_preliminary_models/{0}/{0}_ROC_curve.jpg'.format(model_name))
#         # plot the ROC image
#         img_roc_plot = plt.imshow(img_roc)
#         #remove the axis
#         plt.axis('off')
#         # show the plot
#         plt.show()
#         # creating the html roc auc curve image
#         img_roc_auc = '<img src="'+ 'saved_preliminary_models/{0}/{0}_ROC_curve.jpg'.format(model_name) + '">'
#         return higher_than_50k_probs_auc, img_roc_auc


# # %%
# def confusion_matrix_fuc(model_name,y_train_copy_prep,y_train_copy_pred):
#     #path to save the confusion matrix
#     confusion_matrix_path = Path('saved_preliminary_models/{0}/{0}_confusion_matrix.jpg'.format(model_name))
#     try:
#         #check if the path exists
#         confusion_matrix_path.resolve(strict=True)
#     except FileNotFoundError:
#         print('\n         Confusion Matrix\n')
#         #plot confusion matrix
#         confusion_matrix = ConfusionMatrixDisplay.from_predictions(y_train_copy_prep,y_train_copy_pred, cmap='Blues',values_format='d')
#         #give a title to the plot using the model name
#         plt.title('Confusion Matrix')
#         #save the plot as jpg
#         plt.savefig('saved_preliminary_models/{0}/{0}_confusion_matrix.jpg'.format(model_name))
#         #show the plot
#         plt.show()
#         #img_conf_matrix = 'saved_preliminary_models/{0}/{0}_confusion_matrix.jpg'.format(model_name)
#         img_conf_matrix_html = '<img src="' + 'saved_preliminary_models/{0}/{0}_confusion_matrix.jpg'.format(model_name) + '">'
#         return img_conf_matrix_html
#     else:
#         img_conf_matrix_html = '<img src="' + 'saved_preliminary_models/{0}/{0}_confusion_matrix.jpg'.format(model_name) + '">'
#         img_conf_matrix = mpimg.imread('saved_preliminary_models/{0}/{0}_confusion_matrix.jpg'.format(model_name))
#         # plot the confusion matrix image
#         img_conf_matrix_plot = plt.imshow(img_conf_matrix)
#         # disable the axis
#         plt.axis('off')
#         plt.show()
#         return img_conf_matrix_html

# # %%
# def scores_cal_fuc(model_name,X_train_copy_prep,y_train_copy_prep):
#     score_list = ['precision','recall','f1','accuracy','roc_auc']
#     scores = []
#     scores_mean_for_each_type = []
#     scores_mean = 0
#     scores_std = 0
#     # path to save the model folder
#     model_dir_path = Path('saved_preliminary_models/{0}/'.format(model_name))
#     files_start_with_score_path = []
#     #for loop to check if there is any file start with 'score' in the model folder
#     for i in os.listdir(model_dir_path):
#         if os.path.isfile(os.path.join(model_dir_path,i)) and 'score' in i:
#             files_start_with_score_path.append(os.path.join(model_dir_path,i))
#     # file that start with 'score' found, load the scores list, mean and std using pickle
#     if files_start_with_score_path:
#         for score_type in score_list:
#             # load the scores list
#             with open('saved_preliminary_models/{0}/score_{1}_list.pickle'.format(model_name,score_type),'rb') as handle:
#                 scores = pickle.load(handle)
#             # load the mean score
#             with open('saved_preliminary_models/{0}/score_{1}_mean.pickle'.format(model_name,score_type),'rb') as handle:
#                 scores_mean = pickle.load(handle)
#                 scores_mean_for_each_type.append(scores_mean)
#             # load the std score
#             with open('saved_preliminary_models/{0}/score_{1}_std.pickle'.format(model_name,score_type),'rb') as handle:
#                 scores_std = pickle.load(handle)
#             print('\n                        {} score\n'.format(score_type))
#             print('Scores: {}\n'.format(scores))
#             print('Mean of the scores: {}\n'.format(scores_mean))
#             print('Standard deviation of the scores: {}\n\n'.format(scores_std))
#         return scores_mean_for_each_type
#     # no file start with score in the model folder
#     else:
#         for score_type in score_list:
#             # calculate the scores for each score type using kfold cross validation
#             scores = cross_val_score(model,X_train_copy_prep,y_train_copy_prep,scoring=score_type,cv=10,n_jobs=-1)
#             scores_mean = scores.mean()
#             scores_mean_for_each_type.append(scores_mean)
#             scores_std = scores.std()
#             print('\n                        {} score\n'.format(score_type))
#             print('Scores: {}\n'.format(scores))
#             print('Mean of the scores: {}\n'.format(scores_mean))
#             print('Standard deviation of the scores: {}\n\n'.format(scores_std))
#             # save the scores using pickle
#             with open('saved_preliminary_models/{0}/score_{1}_list.pickle'.format(model_name,score_type),'wb') as handle:
#                 pickle.dump(scores,handle)
#             # save the mean scores using pickle
#             with open('saved_preliminary_models/{0}/score_{1}_mean.pickle'.format(model_name,score_type),'wb') as handle:
#                 pickle.dump(scores_mean,handle)
#             # save the standard deviation scores using pickle
#             with open('saved_preliminary_models/{0}/score_{1}_std.pickle'.format(model_name,score_type),'wb') as handle:
#                 pickle.dump(scores_std,handle)
#         return scores_mean_for_each_type

# # %%
# def classification_report_fuc(model_name,y_train_copy_prep,y_train_copy_pred):
#     # path to save the classification report
#     class_rep_path = Path('saved_preliminary_models/{0}/class_rep_{0}.pickle'.format(model_name))
#     try:
#         #check if the path exists
#         class_rep_path.resolve(strict=True)
#     except FileNotFoundError:
#         #cross validation prediction with kfold = 10
#         print('\n                Classification Report\n')
#         #classification report
#         cls_rep = classification_report(y_train_copy_prep,y_train_copy_pred)
#         print(cls_rep)
#         # save the classification report
#         with open('saved_preliminary_models/{0}/class_rep_{0}.pickle'.format(model_name),'wb') as handle:
#             pickle.dump(cls_rep,handle)
#         return cls_rep
#     else:
#         # if it exist load the classification report
#         with open('saved_preliminary_models/{0}/class_rep_{0}.pickle'.format(model_name),'rb') as handle:
#             cls_rep = pickle.load(handle)
#             print('                       {} Classification Report\n'.format(model_name))
#             print(cls_rep)
#             return cls_rep

# # %%
# def load_coef_and_plot(model_name):
#     with open('saved_preliminary_models/{0}/coef_{0}.pickle'.format(model_name),'rb') as handle:
#         col_with_coef = pickle.load(handle)
#     # print the coefficients of the model
#     # print("\nCoefficients for feature importance:\n")
#     # [print(i) for i in col_with_coef]
#     # print('\n')
#     # load top 10 features image and plot it
#     img_col_with_coef_df_top_10 = mpimg.imread('saved_preliminary_models/{0}/{0}_top_10.jpg'.format(model_name))
#     # plot the confusion matrix image
#     img_col_with_coef_df_top_10_plot = plt.imshow(img_col_with_coef_df_top_10)
#     #remove the axis
#     plt.axis('off')
#     plt.show()
#     # load bottom 10 features image and plot it
#     img_col_with_coef_df_bottom_10 = mpimg.imread('saved_preliminary_models/{0}/{0}_bottom_10.jpg'.format(model_name))
#     # plot the confusion matrix image
#     img_col_with_coef_df_bottom_10_plot = plt.imshow(img_col_with_coef_df_bottom_10)
#     #remove the axis
#     plt.axis('off')
#     plt.show()
#     # save the top 10 features plot to a html tag
#     col_with_coef_df_top_10_html = '<img src="' + 'saved_preliminary_models/{0}/{0}_top_10.jpg'.format(model_name) + '">'
#     # save the bottom 10 features plot to a html tag
#     col_with_coef_df_bottom_10_html = '<img src="' + 'saved_preliminary_models/{0}/{0}_bottom_10.jpg'.format(model_name) + '">'

#     return col_with_coef, col_with_coef_df_top_10_html, col_with_coef_df_bottom_10_html


# # %%
# def fit_and_save_coef_and_plot(model_name, X_train_copy_prep, coef):
#     columns_names = X_train_copy_prep.columns
#     col_with_coef = list(zip(columns_names,coef))
#     col_with_coef.sort(key=lambda x: x[1],reverse=True)
#     # print("\nCoefficients for feature importance:\n")
#     # [print(i) for i in col_with_coef]
#     # print('\n')
#     # horizontal bar plot of the top 10 features
#     col_with_coef_df_top_10 = pd.DataFrame(col_with_coef[:10], columns=['Columns','Coefficients'])
#     # horizontal bar plot of the bottom 10 features
#     col_with_coef_df_bottom_10 = pd.DataFrame(col_with_coef[-10:], columns=['Columns','Coefficients'])
#     sns.barplot(y=col_with_coef_df_top_10['Columns'],x=col_with_coef_df_top_10['Coefficients'])
#     # plot title top 10
#     plt.title('Top 10 most predictive features')
#     # save the plot to a jpg file
#     plt.savefig('saved_preliminary_models/{0}/{0}_top_10.jpg'.format(model_name))
#     plt.show()
#     sns.barplot(y=col_with_coef_df_bottom_10['Columns'],x=col_with_coef_df_bottom_10['Coefficients'])
#     # plot title bottom 10
#     plt.title('Top 10 least predictive features')
#     # save the plot to a jpg file
#     plt.savefig('saved_preliminary_models/{0}/{0}_bottom_10.jpg'.format(model_name))
#     plt.show()
#     # save the coefficients of the model to pickle
#     with open('saved_preliminary_models/{0}/coef_{0}.pickle'.format(model_name),'wb') as handle:
#         pickle.dump(col_with_coef,handle)
#     # save the top 10 features plot to a html tag
#     col_with_coef_df_top_10_html = '<img src="' + 'saved_preliminary_models/{0}/{0}_top_10.jpg'.format(model_name) + '">'
#     # save the bottom 10 features plot to a html tag
#     col_with_coef_df_bottom_10_html = '<img src="' + 'saved_preliminary_models/{0}/{0}_bottom_10.jpg'.format(model_name) + '">'

#     return col_with_coef, col_with_coef_df_top_10_html, col_with_coef_df_bottom_10_html

# # %%
# col_with_coef = []
# def folder_and_file_model_check(model, model_name, X_train_copy_prep, y_train_copy_prep):
#     col_with_coef_df_top_10_html = ''
#     col_with_coef_df_bottom_10_html = ''
#     # check if the folder with the model name exist and if not create them
#     if not os.path.exists('saved_preliminary_models/{}'.format(model_name)):
#         os.makedirs('saved_preliminary_models/{}'.format(model_name))
#     # check if the model file exist and if not create, train and save it
#     model_file_path = Path('saved_preliminary_models/{0}/{0}_model.sav'.format(model_name))
#     try:
#         model_file_path.resolve(strict=True)
#     except FileNotFoundError:
#         model_trn = model.fit(X_train_copy_prep,y_train_copy_prep)
#         joblib.dump(model_trn,model_file_path)
#         # coeficient of the model for feature importance using switch-case statement [new in Python 3.10]
#         match model_name:
#             # for sgd, logistic regression and linear discriminant analysis, use coef_
#             case 'SGD' | 'Logistic_regression' | 'Linear_discriminant_analysis':
#                 coef_of_each_feat = model_trn.coef_[0]
#                 col_with_coef, col_with_coef_df_top_10_html, col_with_coef_df_bottom_10_html = fit_and_save_coef_and_plot(model_name, X_train_copy_prep, coef_of_each_feat)
#             #  no coefficients for the svm model as it took a while to train
#             case 'Support_vector_machine':
#                 # no coefficients or feature importance
#                 col_with_coef = 'No coeficient or feature importance for this model'
#                 pass
#             # for decision tree, random forest, gradient boosting, adaboost and Extra_trees, use feature_importances_
#             case 'Decision_tree' | 'Random_forest' | 'Gradient_boosting' | 'AdaBoost' | 'Extra_trees':
#                 coef_of_each_feat = model_trn.feature_importances_
#                 col_with_coef, col_with_coef_df_top_10_html, col_with_coef_df_bottom_10_html = fit_and_save_coef_and_plot(model_name, X_train_copy_prep, coef_of_each_feat)
#             # does not have does not offer an intrinsic method to evaluate feature importance. refer to https://stackoverflow.com/questions/62933365/how-to-get-the-feature-importance-in-gaussian-naive-bayes, will use permutation_importance
#             case 'Gaussian_naive_bayes':
#                 #Gaussian naive uses the permutation importance method to evaluate feature importance
#                 imps = permutation_importance(model_trn, X_train_copy_prep, y_train_copy_prep)
#                 coef_of_each_feat = imps.importances_mean
#                 col_with_coef, col_with_coef_df_top_10_html, col_with_coef_df_bottom_10_html = fit_and_save_coef_and_plot(model_name, X_train_copy_prep, coef_of_each_feat)
#             # Feature importance is not defined for the KNN Classification algorithm
#             case 'K-Nearest_neighbors':
#                 # no coefficients or feature importance
#                 col_with_coef = 'No coeficient or feature importance for this model'
#                 pass
#             case 'Bagging':
#                 coef_of_each_feat = np.mean([tree.feature_importances_ for tree in model_trn.estimators_], axis=0)
#                 col_with_coef, col_with_coef_df_top_10_html, col_with_coef_df_bottom_10_html = fit_and_save_coef_and_plot(model_name, X_train_copy_prep, coef_of_each_feat)
#             # Feature importance is not defined for the Neural Network Classification algorithm
#             case 'Neural_network':
#                 # no coefficients or feature importance
#                 col_with_coef = 'No coeficient or feature importance for this model'
#                 pass
#     else:
#         # if it exist load the model
#         model_trn = joblib.load(model_file_path)
#         #load the coefficients of the model from pickle
#         match model_name:
#             case 'SGD' | 'Logistic_regression' | 'Linear_discriminant_analysis':
#                 col_with_coef, col_with_coef_df_top_10_html, col_with_coef_df_bottom_10_html = load_coef_and_plot(model_name)
#             case 'Support_vector_machine':
#                 col_with_coef = 'No coeficient or feature importance for this model'
#                 pass
#             case 'Decision_tree' | 'Random_forest' | 'Gradient_boosting' | 'AdaBoost' | 'Extra_trees':
#                 col_with_coef, col_with_coef_df_top_10_html, col_with_coef_df_bottom_10_html = load_coef_and_plot(model_name)
#             case 'Gaussian_naive_bayes':
#                 col_with_coef, col_with_coef_df_top_10_html, col_with_coef_df_bottom_10_html = load_coef_and_plot(model_name)
#             case 'K-Nearest_neighbors':
#                 col_with_coef = 'No coeficient or feature importance for this model'
#                 pass
#             case 'Bagging':
#                 col_with_coef, col_with_coef_df_top_10_html, col_with_coef_df_bottom_10_html = load_coef_and_plot(model_name)
#             case 'Neural_network':
#                 col_with_coef = 'No coeficient or feature importance for this model'
#                 pass


#     # check if y_train_copy_prep exist and if not create it
#     y_train_copy_pred_path = Path('saved_preliminary_models/{0}/y_train_copy_pred_{0}.sav'.format(model_name))
#     try:
#         y_train_copy_pred_path.resolve(strict=True)
#     except FileNotFoundError:
#         #cross validation prediction with kfold = 10
#         y_train_copy_pred = cross_val_predict(model_trn,X_train_copy_prep,y_train_copy_prep,cv=10,n_jobs=-1)
#         #save the predictions
#         joblib.dump(y_train_copy_pred,y_train_copy_pred_path)
#         return y_train_copy_pred, model_trn, col_with_coef, col_with_coef_df_top_10_html, col_with_coef_df_bottom_10_html
#     else:
#         # if it exist load the predictions
#         y_train_copy_pred = joblib.load(y_train_copy_pred_path)
#         return y_train_copy_pred, model_trn, col_with_coef, col_with_coef_df_top_10_html, col_with_coef_df_bottom_10_html

# # %%
# def evaluate_model(model,model_name,X_train_copy_prep,y_train_copy_prep):
#     print('\n       ***************  {}  ***************\n'.format(model_name))
#     #create the folder and the model file if they don't exist
#     y_train_copy_pred,model_trn, col_with_coef, col_with_coef_df_top_10_html, col_with_coef_df_bottom_10_html = folder_and_file_model_check(model,model_name,X_train_copy_prep,y_train_copy_prep)
#     # return the classification report
#     classification_report_fuc(model_name,y_train_copy_prep,y_train_copy_pred)
#     # print the scores by score type, mean scores and std scores and return the mean scores
#     scores_mean= scores_cal_fuc(model_name,X_train_copy_prep,y_train_copy_prep)
#     # return the confusion matrix
#     img_conf_matrix = confusion_matrix_fuc(model_name,y_train_copy_prep,y_train_copy_pred)
#     # return the ROC curve and numpy array of image auc and roc curve
#     auc_score_train, img_roc_auc  = roc_curve_fuc(model_trn,model_name,X_train_copy_prep,y_train_copy_prep)
#     # check if the model has overfit
#     is_overfitting = check_overfit(scores_mean[4],auc_score_train)
#     # create a comparison summary table
#     growing_summary_table_fuc(model_name, scores_mean[0], scores_mean[1], scores_mean[2], scores_mean[3], scores_mean[4], auc_score_train, is_overfitting, img_conf_matrix, img_roc_auc, col_with_coef, col_with_coef_df_top_10_html, col_with_coef_df_bottom_10_html)

# # %% [markdown]
# # ## 5.1 Quick models comparison

# # %%
# model_dict = {
#     'SGD':SGDClassifier(random_state=42,loss='log'),
#     'Logistic_regression':LogisticRegression(random_state=42,max_iter=1000),
#     'Support_vector_machine':SVC(random_state=42,probability=True),
#     'Decision_tree':DecisionTreeClassifier(random_state=42),
#     'Random_forest':RandomForestClassifier(random_state=42),
#     'Gaussian_naive_bayes':GaussianNB(),
#     'K-Nearest_neighbors':KNeighborsClassifier(),
#     'Gradient_boosting':GradientBoostingClassifier(random_state=42),
#     'Linear_discriminant_analysis':LinearDiscriminantAnalysis(),
#     'Bagging':BaggingClassifier(random_state=42),
#     'Neural_network':MLPClassifier(random_state=42,max_iter=1000),
#     'AdaBoost':AdaBoostClassifier(random_state=42),
#     'Extra_trees':ExtraTreesClassifier(random_state=42),
#     }

# # %%
# def model_evaluation_df(model_dict, X_train_copy_prep, y_train_copy_prep):
#     for model_name,model in model_dict.items():
#         evaluate_model(model,model_name,X_train_copy_prep,y_train_copy_prep)
# model_evaluation_df(model_dict,X_train_copy_prep, y_train_copy_prep)

# # %%
# create_summary_table(summary_data)

# # %% [markdown]
# # ## 5.3. Analyze the most significant variables for each algorithm.
# # ### (For each model, use N-fold cross-validation and compute the mean and standard deviation of the performance measure on the N folds)

# # %% [markdown]
# # Done in quick models comparison

# # %% [markdown]
# # ## 5.4 Analyze the types of errors the models make.

# # %% [markdown]
# # The most important metrics here is ***precision***. To explain this, let's say that we want to use this to predict the income of applicant who want to apply for a mortgage. We would care more rejecting some good applicants(Low variance) but keeping a few applicants which are 100% sure that they won't default on their mortgage(High precision).
# # 

# # %% [markdown]
# # ## 5.5 Perform a quick round of feature selection and/or engineering.

# # %% [markdown]
# # Dropping drop the 10 least predictive features for Random forest, Gradiant boosting and Bagging retrain the model

# # %%
# #changing the summary data to a numpy array for easier manipulation
# np_summary_data = np.array(summary_data)

# # %%
# pd.set_option('max_colwidth', 2000)
# model_ft_to_drop_df_all = pd.DataFrame(np_summary_data[:,[0,11]], columns=['Model name','Least predictive feat'])

# # %%
# # extract the most promising model: Random forest, Bagging, Gradient boosting, KNN, and Neural network.
# model_ft_to_drop_df = model_ft_to_drop_df_all.iloc[[4,6,7,9,10]]
# # regular expression to extract the feature names
# patterns = ':\s*([^:.<]+)<'

# # %%
# # extract the feature names as a list from the model_ft_to_drop_df
# def extract_ft_names(model_ft_to_drop_df):
#     for _, row in model_ft_to_drop_df.iterrows():
#         if row['Least predictive feat'] != 'No coeficient or feature importance for this model':
#             row['Least predictive feat'] = re.findall(patterns,row['Least predictive feat'])
# extract_ft_names(model_ft_to_drop_df)

# # %%
# #Implement a feature selection function that takes in the least predictive features for each model, drop them and retrain the models automatically.
# def drop_least_useful_ft(model_name,feat_list):
#     X_train_copy_prep_drop_ft = X_train_copy_prep.drop(feat_list,axis=1)
#     X_train_copy_prep_drop_ft_path = Path('saved_preliminary_models/{0}/X_train_copy_prep_drop_ft_{0}.sav'.format(model_name))
#     try:
#         #check if the path exists
#         X_train_copy_prep_drop_ft_path.resolve(strict=True)
#     except FileNotFoundError:
#         joblib.dump(X_train_copy_prep_drop_ft,X_train_copy_prep_drop_ft_path)
#     model_dict_drop_ft = {model_name:model_dict[model_name]}
#     # retrain each models with the least predictive features dropped
#     model_evaluation_df(model_dict_drop_ft, X_train_copy_prep_drop_ft, y_train_copy_prep)


# # %%
# def drop_ft_retrain(model_ft_to_drop_df):
#     # drop the least predictive features
#     for indexes, row in model_ft_to_drop_df.iterrows():
#         if row['Least predictive feat'] != 'No coeficient or feature importance for this model':
#             #X_train_copy_prep.drop(row['Least predictive feat'],axis=1,inplace=True)
#             #print(row['Model name'],row['Least predictive feat'])
#             drop_least_useful_ft(row['Model name'],row['Least predictive feat'])
# drop_ft_retrain(model_ft_to_drop_df)

# # %% [markdown]
# # ## 5.6. Perform one or two more quick iterations of the five previous steps.

# # %% [markdown]
# # Done!

# # %% [markdown]
# # ## 5.7. Shortlist the top three to five most promising models, preferring models that make different types of errors.
# # 

# # %% [markdown]
# # We will focus on 5 models: Random forest, Bagging, Gradient boosting, KNN, and Neural network as it yields the best results and use precision as the metric.

# # %% [markdown]
# # # 6. Fine-Tune the System

# # %% [markdown]
# # ## 6.1. Fine-tune the hyperparameters using cross-validation

# # %% [markdown]
# # ### 6.1.1 Random Forest

# # %%
# param_grid_rand_for = {
#     'n_estimators' : [100, 200, 300, 500, 800, 1200, 1500, 1800, 2000],
#     'max_features' : ['auto', 'sqrt'],
#     'max_depth' : [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
#     'min_samples_split' : [2, 5, 10, 15, 100],
#     'min_samples_leaf' : [1, 2, 5, 10],
#     'bootstrap' : [True, False]
# }
# rand_forest_clf = model_dict['Random_forest']
# randomize_search_rand_for = RandomizedSearchCV(estimator = rand_forest_clf, param_distributions = param_grid_rand_for, cv=5, n_jobs=-1, verbose=3, scoring='precision')
# X_train_copy_prep_drop_ft = joblib.load('saved_preliminary_models/Random_forest/X_train_copy_prep_drop_ft_Random_forest.sav')
# randomize_search_rand_for.fit(X_train_copy_prep_drop_ft, y_train_copy_prep)

# # %%
# best_rand_for_clf = randomize_search_rand_for.best_estimator_
# best_rand_for_clf.fit(X_train_copy_prep_drop_ft, y_train_copy_prep)
# precision_score_rand_for = np.mean(cross_val_score(best_rand_for_clf,X_train_copy_prep_drop_ft,y_train_copy_prep,scoring='precision',cv=10,n_jobs=-1,verbose=3))

# # %%
# precision_score_rand_for

# # %%
# roc_curve_fuc(best_rand_for_clf, 'Random_forest', X_train_copy_prep_drop_ft, y_train_copy_prep)

# # %%
# randomize_search.best_params_

# # %% [markdown]
# # ### 6.1.2 Neural network

# # %%
# param_grid_nn = {
#     'hidden_layer_sizes': [(1,),(50,),(10,30,10),(20,),(50,50,50), (50,100,50), (100,),(500, 400, 300, 200, 100), (400, 400, 400, 400, 400)],
#     'activation': ['identity', 'logistic', 'tanh', 'relu'],
#     'solver': ['lbfgs','sgd', 'adam'],
#     'alpha': [0.005,0.0005, 0.05],
#     'learning_rate': ['constant','adaptive'],
# }


# # %%
# Neural_network_clf = model_dict['Neural_network']

# # %%
# randomize_search_nn = RandomizedSearchCV(estimator = Neural_network_clf, param_distributions = param_grid_nn, cv=5, n_jobs=-1, verbose=3, scoring='precision')

# # %%
# randomize_search_nn.fit(X_train_copy_prep, y_train_copy_prep)

# # %%
# best_nn_clf = randomize_search_nn.best_estimator_
# best_nn_clf.fit(X_train_copy_prep, y_train_copy_prep)
# precision_score_nn = np.mean(cross_val_score(best_nn_clf,X_train_copy_prep,y_train_copy_prep,scoring='precision',cv=10,n_jobs=-1,verbose=3))
# precision_score_nn

# # %% [markdown]
# # ### 6.1.3 Knn

# # %%
# param_grid_knn = {
#     'n_neighbors' : [5,7,9,11,13,15],
#     'weights' : ['uniform','distance'],
#     'metric' : ['minkowski','euclidean','manhattan']
#     }

# # %%
# knn_clf = model_dict['K-Nearest_neighbors']
# randomize_search_knn = RandomizedSearchCV(estimator = knn_clf, param_distributions = param_grid_knn, cv=5, n_jobs=-1, verbose=3, scoring='precision')
# randomize_search_knn.fit(X_train_copy_prep, y_train_copy_prep)
# best_knn_clf = randomize_search_knn.best_estimator_
# best_knn_clf.fit(X_train_copy_prep, y_train_copy_prep)
# precision_score_knn = np.mean(cross_val_score(best_knn_clf,X_train_copy_prep,y_train_copy_prep,scoring='precision',cv=10,n_jobs=-1,verbose=3))
# precision_score_knn

# # %% [markdown]
# # ### 6.1.4 Gradient_boosting

# # %%
# param_grid_gb = {
#     "n_estimators":[5,50,250,500],
#     "max_depth":[1,3,5,7,9],
#     "learning_rate":[0.01,0.1,1,10,100]
# }

# # %%
# gb_clf = model_dict['Gradient_boosting']
# randomize_search_gb = RandomizedSearchCV(estimator = gb_clf, param_distributions = param_grid_gb, cv=5, n_jobs=-1, verbose=3, scoring='precision')
# X_train_copy_prep_drop_ft = joblib.load('saved_preliminary_models/Gradient_boosting/X_train_copy_prep_drop_ft_Gradient_boosting.sav')
# randomize_search_gb.fit(X_train_copy_prep_drop_ft, y_train_copy_prep)
# best_gb_clf = randomize_search_gb.best_estimator_
# best_gb_clf.fit(X_train_copy_prep_drop_ft, y_train_copy_prep)
# precision_score_gb = np.mean(cross_val_score(best_gb_clf,X_train_copy_prep_drop_ft,y_train_copy_prep,scoring='precision',cv=10,n_jobs=-1,verbose=3))
# precision_score_gb

# # %% [markdown]
# # ### 6.1.5 Bagging

# # %%
# param_grid_bag = {
#     'bootstrap': [True, False],
#     'bootstrap_features': [True, False],
#     'n_estimators': [5, 10, 15],
#     'max_samples' : [0.6, 0.8, 1.0],
#     'max_features' : [0.6, 0.8, 1.0]
# }

# # %%
# bag_clf = model_dict['Bagging']
# randomize_search_bag = RandomizedSearchCV(estimator = bag_clf, param_distributions = param_grid_bag, cv=5, n_jobs=-1, verbose=3, scoring='precision')
# X_train_copy_prep_drop_ft = joblib.load('saved_preliminary_models/Bagging/X_train_copy_prep_drop_ft_Bagging.sav')
# randomize_search_bag.fit(X_train_copy_prep_drop_ft, y_train_copy_prep)
# best_bag_clf = randomize_search_bag.best_estimator_
# best_bag_clf.fit(X_train_copy_prep_drop_ft, y_train_copy_prep)
# precision_score_bag = np.mean(cross_val_score(best_bag_clf,X_train_copy_prep_drop_ft,y_train_copy_prep,scoring='precision',cv=10,n_jobs=-1,verbose=3))
# precision_score_bag

# # %%
# roc_curve_fuc(best_bag_clf, 'Bagging', X_train_copy_prep_drop_ft, y_train_copy_prep)

# %% [markdown]
# ## 6.2 Winner

# %% [markdown]
# We will use the random forest with the best parameters and use the precision as metrics

# %%
# best_parameter_rand_forest = {
#     'n_estimators': 500,
#     'min_samples_split': 10,
#     'min_samples_leaf': 1,
#     'max_features': 'sqrt',
#     'max_depth': 25,
#     'bootstrap': False}

# %% [markdown]
# ## 6.3 Try Ensemble methods. Combining your best models will often produce better performance than running them individually.

# %% [markdown]
# Done!

# %% [markdown]
# ### 6.4 Once you are confident about your final model, measure its performance on the test set to estimate the generalization error.

# # %%
# X_test_copy_prep

# # %%
# X_train_copy_prep_drop_ft

# # %%
# # get the row that are not present in another dataframe
# col_dropped = []
# for col in X_test_copy_prep.columns:
#     if col not in X_train_copy_prep_drop_ft.columns:
#         col_dropped.append(col)

# # %%
# col_dropped

# # %%
# X_test_copy_prep_drop_ft = X_test_copy_prep.drop(col_dropped, axis=1)

# # %%
# X_test_copy_prep_drop_ft.columns

# # %%
# final_predictions = best_rand_for_clf.predict(X_test_copy_prep_drop_ft)

# # %%
# final_predictions

# # %%
# final_predictions.shape

# # %%
# y_test_copy_prep.shape

# # %%
# n_correct = sum(final_predictions == y_test_copy_prep)

# # %%
# print(n_correct/len(final_predictions))

# # %%
# # save the model
# joblib.dump(best_rand_for_clf, 'Random_forest_bst_mdl.sav')

# # %%




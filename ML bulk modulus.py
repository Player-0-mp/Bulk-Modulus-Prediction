#!/usr/bin/env python
# coding: utf-8

# In[1]:


from matminer.datasets.convenience_loaders import load_elastic_tensor
df = load_elastic_tensor()


# In[2]:


df.head()


# In[3]:


df.columns


# In[5]:


unwanted_columns = ["volume", "nsites", "compliance_tensor", "elastic_tensor", 
                    "elastic_tensor_original", "K_Voigt", "G_Voigt", "K_Reuss", "G_Reuss"]
df = df.drop(unwanted_columns, axis=1)


# In[6]:


df.head()


# In[7]:


df.describe()


# In[8]:


from matminer.featurizers.conversions import StrToComposition
df = StrToComposition().featurize_dataframe(df, "formula")
df.head()


# In[9]:


from matminer.featurizers.composition import ElementProperty

ep_feat = ElementProperty.from_preset(preset_name="magpie")
df = ep_feat.featurize_dataframe(df, col_id="composition")  # input the "composition" column to the featurizer
df.head()


# In[10]:


from matminer.featurizers.conversions import CompositionToOxidComposition
from matminer.featurizers.composition import OxidationStates

df = CompositionToOxidComposition().featurize_dataframe(df, "composition")

os_feat = OxidationStates()
df = os_feat.featurize_dataframe(df, "composition_oxid")
df.head()


# In[11]:


from matminer.featurizers.structure import DensityFeatures

df_feat = DensityFeatures()
df = df_feat.featurize_dataframe(df, "structure")  # input the structure column to the featurizer
df.head()


# In[12]:


y = df['K_VRH'].values
excluded = ["G_VRH", "K_VRH", "elastic_anisotropy", "formula", "material_id", 
            "poisson_ratio", "structure", "composition", "composition_oxid"]
X = df.drop(excluded, axis=1)
print("There are {} possible descriptors:\n\n{}".format(X.shape[1], X.columns.values))


# In[13]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

lr = LinearRegression()

lr.fit(X, y)

# get fit statistics
print('training R2 = ' + str(round(lr.score(X, y), 3)))
print('training RMSE = %.3f' % np.sqrt(mean_squared_error(y_true=y, y_pred=lr.predict(X))))


# In[14]:


from sklearn.model_selection import KFold, cross_val_score

# Use 10-fold cross validation (90% training, 10% test)
crossvalidation = KFold(n_splits=10, shuffle=True, random_state=1)
scores = cross_val_score(lr, X, y, scoring='neg_mean_squared_error', cv=crossvalidation, n_jobs=1)
rmse_scores = [np.sqrt(abs(s)) for s in scores]
r2_scores = cross_val_score(lr, X, y, scoring='r2', cv=crossvalidation, n_jobs=1)

print('Cross-validation results:')
print('Folds: %i, mean R2: %.3f' % (len(scores), np.mean(np.abs(r2_scores))))
print('Folds: %i, mean RMSE: %.3f' % (len(scores), np.mean(np.abs(rmse_scores))))


# In[18]:


from figrecipes import PlotlyFig
from sklearn.model_selection import cross_val_predict

pf = PlotlyFig(x_title='DFT (MP) bulk modulus (GPa)',
               y_title='Predicted bulk modulus (GPa)',
               title='Linear regression',
               mode='notebook',
               filename="lr_regression.html")

pf.xy(xy_pairs=[(y, cross_val_predict(lr, X, y, cv=crossvalidation)), ([0, 400], [0, 400])], 
      labels=df['formula'], 
      modes=['markers', 'lines'],
      lines=[{}, {'color': 'black', 'dash': 'dash'}], 
      showlegends=False
     )


# In[19]:


from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=50, random_state=1)

rf.fit(X, y)
print('training R2 = ' + str(round(rf.score(X, y), 3)))
print('training RMSE = %.3f' % np.sqrt(mean_squared_error(y_true=y, y_pred=rf.predict(X))))


# In[20]:


r2_scores = cross_val_score(rf, X, y, scoring='r2', cv=crossvalidation, n_jobs=-1)
scores = cross_val_score(rf, X, y, scoring='neg_mean_squared_error', cv=crossvalidation, n_jobs=-1)
rmse_scores = [np.sqrt(abs(s)) for s in scores]

print('Cross-validation results:')
print('Folds: %i, mean R2: %.3f' % (len(scores), np.mean(np.abs(r2_scores))))
print('Folds: %i, mean RMSE: %.3f' % (len(scores), np.mean(np.abs(rmse_scores))))


# In[21]:


from figrecipes import PlotlyFig

pf_rf = PlotlyFig(x_title='DFT (MP) bulk modulus (GPa)',
                  y_title='Random forest bulk modulus (GPa)',
                  title='Random forest regression',
                  mode='notebook',
                  filename="rf_regression.html")

pf_rf.xy([(y, cross_val_predict(rf, X, y, cv=crossvalidation)), ([0, 400], [0, 400])], 
      labels=df['formula'], modes=['markers', 'lines'],
      lines=[{}, {'color': 'black', 'dash': 'dash'}], showlegends=False)


# In[22]:


from sklearn.model_selection import train_test_split
X['formula'] = df['formula']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
train_formula = X_train['formula']
X_train = X_train.drop('formula', axis=1)
test_formula = X_test['formula']
X_test = X_test.drop('formula', axis=1)

rf_reg = RandomForestRegressor(n_estimators=50, random_state=1)
rf_reg.fit(X_train, y_train)

# get fit statistics
print('training R2 = ' + str(round(rf_reg.score(X_train, y_train), 3)))
print('training RMSE = %.3f' % np.sqrt(mean_squared_error(y_true=y_train, y_pred=rf_reg.predict(X_train))))
print('test R2 = ' + str(round(rf_reg.score(X_test, y_test), 3)))
print('test RMSE = %.3f' % np.sqrt(mean_squared_error(y_true=y_test, y_pred=rf_reg.predict(X_test))))


# In[23]:


from figrecipes import PlotlyFig
pf_rf = PlotlyFig(x_title='Bulk modulus prediction residual (GPa)',
                  y_title='Probability',
                  title='Random forest regression residuals',
                  mode="notebook",
                  filename="rf_regression_residuals.html")

hist_plot = pf_rf.histogram(data=[y_train-rf_reg.predict(X_train), 
                                  y_test-rf_reg.predict(X_test)],
                            histnorm='probability', colors=['blue', 'red'],
                            return_plot=True
                           )
hist_plot["data"][0]['name'] = 'train'
hist_plot["data"][1]['name'] = 'test'
pf_rf.create_plot(hist_plot)


# In[24]:


#important features used by random forest model
importances = rf.feature_importances_
# included = np.asarray(included)
included = X.columns.values
indices = np.argsort(importances)[::-1]

pf = PlotlyFig(y_title='Importance (%)',
               title='Feature by importances',
               mode='notebook',
               fontsize=20,
               ticksize=15)

pf.bar(x=included[indices][0:10], y=importances[indices][0:10])


# In[ ]:





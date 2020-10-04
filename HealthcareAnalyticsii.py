from fastai.tabular import *

df_train = pd.read_csv('/content/drive/My Drive/Janatahack_Healthcare_AnalyticsII/train.csv')
df_test = pd.read_csv('/content/drive/My Drive/Janatahack_Healthcare_AnalyticsII/test.csv')
df_sub = pd.read_csv('/content/drive/My Drive/Janatahack_Healthcare_AnalyticsII/sample_submission.csv')

############ df_train.drop(['Hospital_type_code','City_Code_Hospital','Hospital_region_code','City_Code_Patient','patientid'], inplace= True, axis = 1)
######### df_test.drop(['Hospital_type_code','City_Code_Hospital','Hospital_region_code','City_Code_Patient','patientid'], inplace= True, axis = 1)


# target = "Stay"

# df_train.Hospital_code = df_train.Hospital_code.astype(object)
# df_train.City_Code_Hospital = df_train.City_Code_Hospital.astype(object)
# df_train.City_Code_Patient = df_train.City_Code_Patient.astype(object)

# df_train['Bed Grade'].fillna(5, inplace = True)
# df_train['City_Code_Patient'].fillna(39, inplace = True)

# df_test['Bed Grade'].fillna(5, inplace = True)
# df_test['City_Code_Patient'].fillna(39, inplace = True)

# path = '/content/drive/My Drive/Janatahack_Healthcare_AnalyticsII/Models'
# pd.set_option('display.max_columns', None)

df_train['Hospital'] = df_train['Hospital_code'].astype(str) + df_train['Hospital_type_code'].astype(str) + df_train['City_Code_Hospital'].astype(str) + df_train['Hospital_region_code'].astype(str)
df_train.drop(['Hospital_code','Hospital_type_code','City_Code_Hospital','Hospital_region_code','patientid','City_Code_Patient'], inplace= True, axis = 1)
df_train['Bed Grade'] = df_train['Bed Grade'].astype('int64')


df_test['Hospital'] = df_test['Hospital_code'].astype(str) + df_test['Hospital_type_code'].astype(str) + df_test['City_Code_Hospital'].astype(str) + df_test['Hospital_region_code'].astype(str)
df_test.drop(['Hospital_code','Hospital_type_code','City_Code_Hospital','Hospital_region_code', 'patientid','City_Code_Patient'], inplace= True, axis = 1)
df_test['Bed Grade'] = df_test['Bed Grade'].astype('int64')

df_train['Age'] = df_train['Age'].astype('category')
df_train['Available Extra Rooms in Hospital'] = df_train['Available Extra Rooms in Hospital'].astype('category')
df_train['Bed Grade'] = df_train['Bed Grade'].astype('category')
df_train['Visitors with Patient'] = df_train['Visitors with Patient'].astype('category')


df_test['Age'] = df_test['Age'].astype('category')
df_test['Available Extra Rooms in Hospital'] = df_test['Available Extra Rooms in Hospital'].astype('category')
df_test['Bed Grade'] = df_test['Bed Grade'].astype('category')
df_test['Visitors with Patient'] = df_test['Visitors with Patient'].astype('category')

# df_train['PatientCity_AdmissionType'] = df_train.groupby('City_Code_Patient')['Type of Admission'].transform('count')
# df_train['Age_visitors'] = df_train.groupby('Age')['Visitors with Patient'].transform('sum')
# df_train['Department_AdmissionType'] = df_train.groupby('Department')['Type of Admission'].transform('count')
# df_train['wardType_wardFacility'] = df_train.groupby(['Ward_Type','Ward_Facility_Code'])['Ward_Type'].transform('count')
# df_train['hospitalCode_availableBed'] = df_train.groupby(['Hospital_code','Available Extra Rooms in Hospital'])['Available Extra Rooms in Hospital'].transform('sum')
# df_train['age_department'] = df_train.groupby(['Age','Department'])['Department'].transform('count')

# df_test['PatientCity_AdmissionType'] = df_test.groupby('City_Code_Patient')['Type of Admission'].transform('count')
# df_test['Age_visitors'] = df_test.groupby('Age')['Visitors with Patient'].transform('sum')
# df_test['Department_AdmissionType'] = df_test.groupby('Department')['Type of Admission'].transform('count')
# df_test['wardType_wardFacility'] = df_test.groupby(['Ward_Type','Ward_Facility_Code'])['Ward_Type'].transform('count')
# df_test['hospitalCode_availableBed'] = df_test.groupby(['Hospital_code','Available Extra Rooms in Hospital'])['Available Extra Rooms in Hospital'].transform('sum')
# df_test['age_department'] = df_test.groupby(['Age','Department'])['Department'].transform('count')

#Combining train and test data
df_combined = pd.concat([df_train, df_test], axis=0)
#df_combined

# Feature Engineering
df_combined = df_combined.replace({
    'Department':{'surgery': 1,'TB & Chest disease' : 2, 'radiotherapy': 3, 'anesthesia' : 4, 'gynecology': 5},
    'Severity of Illness':{'Minor': 1,'Moderate' : 2, 'Extreme' : 3},
    'Type of Admission':{'Urgent': 1,'Emergency' : 2, 'Trauma' : 3}})

# df_combined['hospital_age_department'] = df_combined.groupby(['Hospital', 'Age','Department'])['Stay'].transform('count')
# df_combined['hospital_age_visitors'] = df_combined.groupby(['Hospital', 'Age','Visitors with Patient'])['Stay'].transform('count')
# df_combined['age_WardFacility_WardType'] = df_combined.groupby(['Age', 'Ward_Facility_Code','Ward_Type'])['Stay'].transform('count')

df_combined.Department = df_combined.Department.astype('int64')
df_combined['Severity of Illness'] = df_combined['Severity of Illness'].astype('category')
df_combined['Type of Admission'] = df_combined['Type of Admission'].astype('int64')

# Splitting into train and test

df_train = df_combined[(df_combined['case_id'] <318439 )]
df_test = df_combined[(df_combined['case_id'] >= 318439 )]
df_test.drop(['Stay'], axis=1, inplace= True)

df_train.drop(['case_id'], axis = 1, inplace= True)
df_test.drop(['case_id'], axis = 1, inplace= True)

df_train.info()

df_train['Visitors with Patient'] = df_train['Visitors with Patient'].astype('category')
df_test['Visitors with Patient'] = df_test['Visitors with Patient'].astype('category')

#The dependent variable/target
dep_var = 'Stay'

#The list of categorical features in the dataset
cat_names = ['Ward_Type','Ward_Facility_Code', 'Age', 'Bed Grade','Hospital'] 

#The list of continuous features in the dataset
#cont_names =['Admission_Deposit','Available Extra Rooms in Hospital', 'Visitors with Patient','PatientCity_AdmissionType','Age_visitors','Department_AdmissionType','wardType_wardFacility','hospitalCode_availableBed','age_department'] 
#cont_names =['Admission_Deposit','Available Extra Rooms in Hospital', 'Visitors with Patient','hospital_age_department','hospital_age_visitors','age_WardFacility_WardType'] 
cont_names =['Admission_Deposit'] 

#List of Processes/transforms to be applied to the dataset
procs = [FillMissing, Categorify, Normalize]

#Start index for creating a validation set from train_data
start_indx = len(df_train) - int(len(df_train) * 0.2)

#End index for creating a validation set from train_data
end_indx = len(df_train)

#TabularList for Validation
val = (TabularList.from_df(df_train.iloc[start_indx:end_indx].copy(), path=path, cat_names=cat_names, cont_names=cont_names))

test = (TabularList.from_df(df_test, path=path, cat_names=cat_names, cont_names=cont_names, procs=procs))

#TabularList for training
data = (TabularList.from_df(df_train, path=path, cat_names=cat_names, cont_names=cont_names, procs=procs)
                           .split_by_idx(list(range(start_indx,end_indx)))
                           .label_from_df(cols=dep_var)
                           .add_test(test)
                           .databunch())

data.show_batch(rows = 10)

#Initializing the network
learn = tabular_learner(data, layers=[1000,300,100, 50], metrics=accuracy, emb_drop=0.1, callback_fns=ShowGraph)

# select the appropriate learning rate
learn.lr_find()

# we typically find the point where the slope is steepest
learn.recorder.plot(suggestion= True)

# Fit the model based on selected learning rate
learn.fit_one_cycle(3, 3.6e-02,  moms=(0.8,0.7))

# Analyse our model
learn.model
learn.recorder.plot_losses()
learn.show_results()

learn.show_results(rows= 500)

# Predict our target value
predictions, *_ = learn.get_preds(DatasetType.Test)
labels = np.argmax(predictions,1)

label_df = pd.DataFrame(labels)
#label_df1 = pd.DataFrame(label_df.idxmax(axis = 1 ))
label_df.columns = ['Stay']
label_df = label_df.replace({
      'Stay':{0:'0-10', 1:'11-20',2: '21-30', 3:'31-40', 4:'41-50', 5:'51-60', 6:'61-70', 7:'71-80', 8:'81-90', 9:'91-100', 10:'More than 100 Days'}
    })

df_sub['Stay'] = label_df['Stay']
df_sub.to_csv("/content/drive/My Drive/Janatahack_Healthcare_AnalyticsII/fastai_11.csv", header=True, index = False)

#%reset

###################################################################################################################################################################

from fastai.tabular import *

df_train = pd.read_csv('/content/drive/My Drive/Janatahack_Healthcare_AnalyticsII/train.csv')
df_test = pd.read_csv('/content/drive/My Drive/Janatahack_Healthcare_AnalyticsII/test.csv')
df_sub = pd.read_csv('/content/drive/My Drive/Janatahack_Healthcare_AnalyticsII/sample_submission.csv')

target = "Stay"

df_train.info()

df_train.Hospital_code = df_train.Hospital_code.astype(object)
df_train.City_Code_Hospital = df_train.City_Code_Hospital.astype(object)
df_train.City_Code_Patient = df_train.City_Code_Patient.astype(object)

df_test.Hospital_code = df_test.Hospital_code.astype(object)
df_test.City_Code_Hospital = df_test.City_Code_Hospital.astype(object)
df_test.City_Code_Patient = df_test.City_Code_Patient.astype(object)

df_train['Bed Grade'].fillna(5, inplace = True)
df_train['City_Code_Patient'].fillna(39, inplace = True)

df_test['Bed Grade'].fillna(5, inplace = True)
df_test['City_Code_Patient'].fillna(39, inplace = True)

path = '/content/drive/My Drive/Janatahack_Healthcare_AnalyticsII/Models'
pd.set_option('display.max_columns', None)

#The dependent variable/target
dep_var = 'Stay'

#The list of categorical features in the dataset
cat_names = ['Department','Ward_Type','Ward_Facility_Code', 'Type of Admission','Severity of Illness', 'Age', 'Bed Grade','Hospital_code','City_Code_Hospital','City_Code_Patient'] 

#The list of continuous features in the dataset
cont_names =['Admission_Deposit','Available Extra Rooms in Hospital', 'Visitors with Patient'] 

#List of Processes/transforms to be applied to the dataset
procs = [Categorify, Normalize]

#Start index for creating a validation set from train_data
start_indx = len(df_train) - int(len(df_train) * 0.2)

#End index for creating a validation set from train_data
end_indx = len(df_train)

#TabularList for Validation
val = (TabularList.from_df(df_train.iloc[start_indx:end_indx].copy(), path=path, cat_names=cat_names, cont_names=cont_names))

test = (TabularList.from_df(df_test, path=path, cat_names=cat_names, cont_names=cont_names, procs=procs))

#TabularList for training
data = (TabularList.from_df(df_train, path=path, cat_names=cat_names, cont_names=cont_names, procs=procs)
                           .split_by_idx(list(range(start_indx,end_indx)))
                           .label_from_df(cols=dep_var)
                           .add_test(test)
                           .databunch())

data.show_batch(rows = 10)

#Initializing the network
learn = tabular_learner(data, layers=[1000,300, 100, 50], metrics=accuracy, emb_drop=0.1, callback_fns=ShowGraph)

# select the appropriate learning rate
learn.lr_find()

# we typically find the point where the slope is steepest
learn.recorder.plot(suggestion= True)

# Fit the model based on selected learning rate
learn.fit_one_cycle(10, 2.5e-03,  moms=(0.8,0.7))

# Analyse our model
learn.model
learn.recorder.plot_losses()
learn.show_results()

# Predict our target value
predictions, *_ = learn.get_preds(DatasetType.Test)
labels = np.argmax(predictions,1)

label_df = pd.DataFrame(labels)
#label_df1 = pd.DataFrame(label_df.idxmax(axis = 1 ))
label_df.columns = ['Stay']
label_df = label_df.replace({
      'Stay':{0:'0-10', 1:'11-20',2: '21-30', 3:'31-40', 4:'41-50', 5:'51-60', 6:'61-70', 7:'71-80', 8:'81-90', 9:'91-100', 10:'More than 100 Days'}
    })

df_sub['Stay'] = label_df['Stay']
df_sub.to_csv("/content/drive/My Drive/Janatahack_Healthcare_AnalyticsII/fastai_13.csv", header=True, index = False)

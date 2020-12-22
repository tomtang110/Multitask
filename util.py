import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import binarize
import numpy as np
from torch.nn import functional as F
column_names = ['age', 'class_worker', 'det_ind_code', 'det_occ_code', 'education', 'wage_per_hour', 'hs_college',
                    'marital_stat', 'major_ind_code', 'major_occ_code', 'race', 'hisp_origin', 'sex', 'union_member',
                    'unemp_reason', 'full_or_part_emp', 'capital_gains', 'capital_losses', 'stock_dividends',
                    'tax_filer_stat', 'region_prev_res', 'state_prev_res', 'det_hh_fam_stat', 'det_hh_summ',
                    'instance_weight', 'mig_chg_msa', 'mig_chg_reg', 'mig_move_reg', 'mig_same', 'mig_prev_sunbelt',
                    'num_emp', 'fam_under_18', 'country_father', 'country_mother', 'country_self', 'citizenship',
                    'own_or_self', 'vet_question', 'vet_benefits', 'weeks_worked', 'year', 'income_50k']
categorical_columns = ['class_worker', 'det_ind_code', 'det_occ_code', 'education', 'hs_college', 'major_ind_code',
                           'major_occ_code', 'race', 'hisp_origin', 'sex', 'union_member', 'unemp_reason',
                           'full_or_part_emp', 'tax_filer_stat', 'region_prev_res', 'state_prev_res', 'det_hh_fam_stat',
                           'det_hh_summ', 'mig_chg_msa', 'mig_chg_reg', 'mig_move_reg', 'mig_same', 'mig_prev_sunbelt',
                           'fam_under_18', 'country_father', 'country_mother', 'country_self', 'citizenship',
                           'vet_question']
Ignore_columns = ['income_50k', 'marital_stat']
numeric_cols = list(set(column_names) - set(categorical_columns) - set(Ignore_columns))

SEED = 1
import torch

def data_prepare(config):

    def deal_data(label_cols,data):
        label = []
        for lal in label_cols:
            # label.append(to_categorical(data[lal].values,n_class=2))
            label.append(data[lal].values)
        return data.drop(label_cols,axis=1).values,label
    train_df = pd.read_csv(
        config.train_path,
        delimiter=',',
        header=None,
        index_col=None,
        names=column_names
    )
    other_df = pd.read_csv(
        config.test_path,
        delimiter=',',
        header=None,
        index_col=None,
        names=column_names
    )

    train_df['income_50k'] = train_df['income_50k'].apply(lambda x:int(x == ' 50000+.'))
    train_df['marital_stat'] = train_df['marital_stat'].apply(lambda x: int(x == ' Never married'))

    other_df['income_50k'] = other_df['income_50k'].apply(lambda x:int(x == ' 50000+.'))
    other_df['marital_stat'] = other_df['marital_stat'].apply(lambda x: int(x == ' Never married'))


    transformed_train = pd.get_dummies(train_df)
    transformed_other = pd.get_dummies(other_df)

    # Filling the missing column in the other set
    transformed_other['det_hh_fam_stat_ Grandchild <18 ever marr not in subfamily'] = 0

    # One-hot encoding categorical labels
    valid_data,test_data = train_test_split(transformed_other,test_size=0.5,random_state=1)

    train_data,train_label = deal_data(config.label_columns,transformed_train)
    valid_data, valid_label = deal_data(config.label_columns, valid_data)
    test_data, test_label = deal_data(config.label_columns, test_data)


    config.num_feature = train_data.shape[1]
    def transform2content(data,label):
        content = []
        for i in range(len(data)):
            content.append([data[i],label[0][i],label[1][i]])
        return content
    trainset = transform2content(train_data,train_label)
    testset = transform2content(test_data,test_label)
    devset = transform2content(valid_data, valid_label)
    return trainset,testset,devset

def gen_feat_dict(train_df,other_df):
    feat_dict = {}
    df = pd.concat([train_df,other_df],axis=0)
    feat_index_dim = 0
    for col in df.columns:
        if col in Ignore_columns:
            continue
        if col in numeric_cols:
            feat_dict[col] = feat_index_dim
            feat_index_dim += 1
        else:
            single_cate = df[col].unique()
            feat_dict[col] = dict(zip(single_cate,range(feat_index_dim,len(single_cate)+feat_index_dim)))
            feat_index_dim += len(single_cate)

    return feat_dict,feat_index_dim

def df_parse(dfi,feat_dict):
    income_label = (dfi.income_50k == ' 50000+.').astype(int)
    marital_label = (dfi.marital_stat == ' Never married').astype(int)

    dfi = dfi.drop(columns=Ignore_columns)
    # dfi for feature index
    # dfv for feature value which can be either binary (1/0) or float (e.g., 10.24)
    dfv = dfi.copy()
    for col in dfi.columns:
        if col in Ignore_columns:
            dfi.drop(col,axis=1,inplace=True)
            dfv.drop(col,axis=1,inplace=True)
            continue
        if col in numeric_cols:
            dfi[col] = feat_dict[col]
        else:
            dfi[col] = dfi[col].map(feat_dict[col])
            dfv[col] = 1

    xi = dfi.values.tolist()
    xv = dfv.values.tolist()

    content = []
    for i in range(len(xi)):
        content.append([xi[i],xv[i],income_label.values[i],marital_label.values[i]])
    return content



def data_prepare_advanced(config):
    train_df = pd.read_csv(
        config.train_path,
        delimiter=',',
        header=None,
        index_col=None,
        names=column_names
    )
    other_df = pd.read_csv(
        config.test_path,
        delimiter=',',
        header=None,
        index_col=None,
        names=column_names
    )
    feat_dict,feat_dim = gen_feat_dict(train_df,other_df)

    config.num_feature = feat_dim
    config.field_size = len(categorical_columns) + len(numeric_cols)

    val_data,test_data = train_test_split(other_df,test_size=0.5,random_state=1)
    train_set = df_parse(train_df,feat_dict)
    test_set = df_parse(test_data,feat_dict)
    dev_set = df_parse(val_data,feat_dict)
    return train_set,test_set,dev_set






class Dataiterater(object):
    def __init__(self,batches,batch_size,device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size

        self.residue =False
        if len(batches)%batch_size != 0:
            self.residue = True
        self.index = 0
        self.device = device
    def _to_n_tensor(self, datas):

        x = torch.FloatTensor([_[0] for _ in datas]).to(self.device)
        y1 = torch.FloatTensor([_[1] for _ in datas]).to(self.device)
        y2 = torch.FloatTensor([_[2] for _ in datas]).to(self.device)

        return x, (y1,y2)

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_n_tensor(batches)

            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_n_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches

def build_iterator(dataset, config):
    iter = Dataiterater(dataset, config.batch_size, config.device)
    return iter



class Dataiterater_advanced(object):
    def __init__(self,batches,batch_size,device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size

        self.residue =False
        if len(batches)%batch_size != 0:
            self.residue = True
        self.index = 0
        self.device = device
    def _to_n_tensor(self, datas):

        xi = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        xv = torch.FloatTensor([_[1] for _ in datas]).to(self.device)
        y1 = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        y2 = torch.LongTensor([_[3] for _ in datas]).to(self.device)

        return (xi,xv), (y1,y2)

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_n_tensor(batches)

            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_n_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches

def build_iterator_advanced(dataset, config):
    iter = Dataiterater_advanced(dataset, config.batch_size, config.device)
    return iter

def to_categorical(labels,n_class=2):
    batch = labels.shape[0]
    result = np.zeros((batch,n_class))
    for i in range(batch):
        result[i][labels[i]] = 1
    return result

def loss_fn(name):
    if name == 'multiclass':
        return F.cross_entropy
    elif name == 'binary':
        return F.binary_cross_entropy
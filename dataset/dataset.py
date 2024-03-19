import pandas as pd
import numpy as np
import torch
import json
from torch.utils.data import Dataset
from sklearn.model_selection import KFold

lab_event_list = sorted(['Glucose', 'Potassium', 'Sodium', 'Chloride', 'Creatinine',
'Urea Nitrogen', 'Bicarbonate', 'Anion Gap', 'Hemoglobin', 'Hematocrit',
'Magnesium', 'Platelet Count', 'Phosphate', 'White Blood Cells',
'Calcium, Total', 'MCH', 'Red Blood Cells', 'MCHC', 'MCV', 'RDW', 
'Neutrophils', 'Vancomycin'])

class EHRData(Dataset):
    def __init__(self, path='data/processed', split='train'):
        self.path = path
        self.split = split
        self.records = pd.read_csv(f'{path}/valid_records.csv')
        self.labs = pd.read_csv(f'{path}/labs_norm.csv', dtype={'order_provider_id': str, 'value': str, 'comments': str})
        self.records = self.records.merge(self.labs[['subject_id','hadm_id']].drop_duplicates(), on=['subject_id','hadm_id'])
        
        self.all_records = self.records[['subject_id','hadm_id']].drop_duplicates().reset_index(drop=True)
        self.patients = self.all_records[['subject_id']].drop_duplicates().reset_index(drop=True)
        self.gen_folds()
        self.set_fold(0)
        self.set_split(split)

    def __len__(self):
        return len(self.records)
    
    def __getitem__(self, idx):
        item = self.records.iloc[idx]
        subject_id = item['subject_id']
        hadm_id = item['hadm_id']

        labs = self.labs[(self.labs['subject_id'] == subject_id) & 
                         (self.labs['hadm_id'] == hadm_id)].sort_values(['label', 'charttime'])
        
        labs = labs.groupby('label').apply(self.get_gen_mask())
        data = labs.pivot_table(index='label', columns='charttime', values='valuenorm_zscore', fill_value=np.nan).sort_index(axis=1)
        target_mask = labs.pivot_table(index='label', columns='charttime', values='mask', fill_value=0).sort_index(axis=1)
        for var in lab_event_list:
            if var not in data.index:
                data = pd.concat([data, pd.DataFrame(index=[var], columns=data.columns, data=np.nan)])
                target_mask = pd.concat([target_mask, pd.DataFrame(index=[var], columns=target_mask.columns, data=0)])
        data = data.loc[lab_event_list].values.T
        target_mask = target_mask.loc[lab_event_list].values.T
        missing_mask = ~np.isnan(data)
        data = np.nan_to_num(data, nan=0)
        
        data = torch.tensor(data).float()
        missing_mask = torch.tensor(missing_mask).int()
        target_mask = torch.tensor(target_mask).int()
        return data, missing_mask, target_mask
    
    def gen_folds(self):
        self.folds = []
        kf = KFold(n_splits=5)
        kf2 = KFold(n_splits=2)
        for train_index, test_index in kf.split(self.patients):
            test_patients = self.patients.iloc[test_index]
            val_index, test_index_ = next(kf2.split(test_patients))
            self.folds.append((train_index, val_index, test_index_))
    
    def set_fold(self, fold):
        self.fold = fold
        self.train_index, self.val_index, self.test_index = self.folds[fold]
        self.train_records = self.all_records.iloc[self.train_index]
        self.val_records = self.all_records.iloc[self.val_index]
        self.test_records = self.all_records.iloc[self.test_index]
    
    def set_split(self, split):
        self.split = split
        if split == 'train':
            self.records = self.train_records.copy()
        elif split == 'val':
            self.records = self.val_records.copy()
        elif split == 'test':
            self.records = self.test_records.copy()
        elif split == 'all':
            self.records = self.all_records.copy()
        else:
            raise ValueError('Invalid split')
    
    def get_collate():
        def collate_fn(batch):
            data, missing_mask, target_mask = zip(*batch)
            length = [len(x) for x in data]
            max_length = max(length)
            data = [torch.cat([x, torch.zeros(max_length - len(x), x.size(1))]) for x in data]
            missing_mask = [torch.cat([x, torch.zeros(max_length - len(x), x.size(1), dtype=torch.int32)]) for x in missing_mask]
            target_mask = [torch.cat([x, torch.zeros(max_length - len(x), x.size(1), dtype=torch.int32)]) for x in target_mask]
            data = torch.stack(data)
            missing_mask = torch.stack(missing_mask)
            target_mask = torch.stack(target_mask)
            return data, missing_mask, target_mask
        return collate_fn

    def get_gen_mask(self, p=0.15):
        def gen_mask(data):
            length = len(data)
            mask_num = max(1, int(p * length))
            mask_idx = np.random.choice(length, mask_num, replace=False)
            mask = torch.zeros(length)
            mask[mask_idx] = 1
            data['mask'] = mask
            return data
        return gen_mask

imp_event_list = ['wbc','bun','sodium','pt','inr','ptt','platelet','lactate',
                  'hemoglobin','glucose','chloride','creatinine','aniongap',
                  'bicarbonate','hematocrit','heartrate','resprate','tempc',
                  'meanbp','gcs_min','urineoutput','sysbp','diasbp','spo2',
                  'Magnesium','C-reactive protein','bands']

class ImpData(Dataset):
    def __init__(self, path='data/imputation_data', split='train'):
        self.path = path
        self.split = split
        self.records = pd.read_csv(f'{path}/patient_table.csv')
        with open(f'{path}/feature_dict/feature_ms_dict.json', 'r') as f:
            self.feature_ms_dict = json.load(f)
        
        self.all_records = self.records.copy()
        self.gen_folds()
        self.set_fold(0)
        self.set_split(split)

    def __len__(self):
        return len(self.records)
    
    def __getitem__(self, idx):
        item = self.records.iloc[idx]['patient_id']
        data, gt = pd.read_csv(f'{self.path}/train_with_missing/{item}.csv'), pd.read_csv(f'{self.path}/train_groundtruth/{item}.csv')
        data = data.apply(lambda x: x if x.name == 'time' 
                                      else (x - self.feature_ms_dict[x.name][0])/self.feature_ms_dict[x.name][1], 
                          axis=0)
        gt = gt.apply(lambda x: x if x.name == 'time' 
                                  else (x - self.feature_ms_dict[x.name][0])/self.feature_ms_dict[x.name][1], 
                          axis=0)
        data_mask = 1 - data.isna().astype(int).values
        data = data.fillna(0).values
        gt_mask = 1 - gt.isna().astype(int).values
        gt = gt.fillna(0).values
        target_mask = gt_mask - data_mask
        assert target_mask.min() >= 0

        data = torch.tensor(data).float()
        data_mask = torch.tensor(gt_mask).int()
        target_mask = torch.tensor(target_mask).int()
        gt = torch.tensor(gt).float()
        return data[:,1:], data_mask[:,1:], target_mask[:,1:], gt[:,1:]

    def gen_folds(self):
        self.folds = []
        kf = KFold(n_splits=5)
        kf2 = KFold(n_splits=2)
        for train_index, test_index in kf.split(self.all_records):
            test_patients = self.all_records.iloc[test_index]
            val_index, test_index_ = next(kf2.split(test_patients))
            self.folds.append((train_index, val_index, test_index_))
    
    def set_fold(self, fold):
        self.fold = fold
        self.train_index, self.val_index, self.test_index = self.folds[fold]
        self.train_records = self.all_records.iloc[self.train_index]
        self.val_records = self.all_records.iloc[self.val_index]
        self.test_records = self.all_records.iloc[self.test_index]
    
    def set_split(self, split):
        self.split = split
        if split == 'train':
            self.records = self.train_records.copy()
        elif split == 'val':
            self.records = self.val_records.copy()
        elif split == 'test':
            self.records = self.test_records.copy()
        elif split == 'all':
            self.records = self.all_records.copy()
        else:
            raise ValueError('Invalid split')
    
    def get_collate():
        def collate_fn(batch):
            data, data_mask, target_mask, gt = zip(*batch)
            length = [len(x) for x in data]
            max_length = max(length)

            data = [torch.cat([x, torch.zeros(max_length - len(x), x.size(1))]) for x in data]
            data_mask = [torch.cat([x, torch.zeros(max_length - len(x), x.size(1), dtype=torch.int32)]) for x in data_mask]
            target_mask = [torch.cat([x, torch.zeros(max_length - len(x), x.size(1), dtype=torch.int32)]) for x in target_mask]
            gt = [torch.cat([x, torch.zeros(max_length - len(x), x.size(1))]) for x in gt]

            data = torch.stack(data)
            data_mask = torch.stack(data_mask)
            target_mask = torch.stack(target_mask)
            gt = torch.stack(gt)
            return data, data_mask, target_mask, gt
        return collate_fn
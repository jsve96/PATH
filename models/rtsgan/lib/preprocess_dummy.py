import numpy as np
import pandas as pd 
from missingprocessor import Processor

data_path = "../TimeGAN/data"
loc = "stock"
seq_len = 24
df = pd.read_csv('{}/{}_data.csv'.format(data_path,loc), sep = ",")
types = ["continuous" for i in range(len(df.columns))]

P = Processor(types)
# Flip the data to make chronological data
ori_data = P.fit_transform(df)
ori_data = ori_data[::-1]

temp_data = [ori_data[i:i + seq_len] for i in range(0, len(ori_data) - seq_len)]    


from fastNLP import DataSet

dataset = DataSet({"seq_len": [seq_len] * len(temp_data), "dyn": temp_data, "sta":[0]*len(temp_data)})
dic = {
    "train_set": dataset,
    "dynamic_processor": P,
    "static_processor": Processor([])
}
print(P.dim, len(temp_data))



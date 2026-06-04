import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import os

def sine_data_generation (no, seq_len, dim):
  """Sine data generation.
  
  Args:
    - no: the number of samples
    - seq_len: sequence length of the time-series
    - dim: feature dimensions
    
  Returns:
    - data: generated data
  """  
  # Initialize the output
  seed = 0
  np.random.seed(seed)
  data = list()

  # Generate sine data
  for i in range(no):      
    # Initialize each time-series
    temp = list()
    # For each feature
    for _ in range(dim):
      # Randomly drawn frequency and phase
      freq = np.random.uniform(0, 0.1)            
      phase = np.random.uniform(0, 0.1)
          
      # Generate sine signal based on the drawn frequency and phase
      temp_data = [np.sin(freq * j + phase) for j in range(seq_len)] 
      temp.append(temp_data)
        
    # Align row/column
    temp = np.transpose(np.asarray(temp))        
    # Normalize to [0,1]
    temp = (temp + 1)*0.5
    # Stack the generated data
    data.append(np.array(temp))

  return np.stack(data,axis=0)

def real_data_loading(ori_data, seq_len,seed=0):
    """Prepare chronological windowed sequences from raw tabular time series.

    Parameters
    ----------
    ori_data : np.ndarray
        Raw data with shape ``(time, features)``.
    seq_len : int
        Sliding-window sequence length.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Shuffled windows of shape ``(n_windows, seq_len, features)``, maxima, and minima.
    """

    # Flip the data to make chronological data
    L, d = ori_data.shape
    ori_data = ori_data[::-1]
    # Normalize the data
    ori_data, max_, min_ = MinMaxScaler(ori_data)

    # Preprocess the dataset
    temp_data = np.zeros((L - seq_len, seq_len, d))
    # Cut data by sequence length
    for i in range(0, len(ori_data) - seq_len):
        temp_data[i] = ori_data[i:i + seq_len]

    # Mix the datasets (to make it similar to i.i.d)
    np.random.seed(seed)
    idx = np.random.permutation(len(temp_data))
    return temp_data[idx], max_, min_


def MinMaxScaler(data):
    """Apply feature-wise min-max scaling.

    Parameters
    ----------
    data : np.ndarray
        Input array.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Normalized data, per-feature maxima, and per-feature minima.
    """
    min_, max_ = np.min(data, 0), np.max(data, 0)
    numerator = data - min_
    denominator = max_ - min_
    norm_data = numerator / (denominator + 1e-7)
    return norm_data, max_, min_





if __name__ == '__main__':
  
    sine_data = sine_data_generation(10000,24,5)
    print('Generated Sine Data')
    print(f'Total Samepls: {sine_data.shape}')
    sine_train, sine_test = train_test_split(sine_data,test_size=0.2,random_state=0)
    print(f'Train Samples: {sine_train.shape}')
    print(f'saved at data/sine/train.npy')
    np.save('./data/sine/train.npy',sine_train)

    print(f'Test Samples: {sine_test.shape}')
    print(f'saved at data/sine/test.npy')
    np.save('./data/sine/test.npy',sine_test)


    ### AIR 
    air = pd.read_excel("./data/AirQuality.xlsx").values[:,2:]
    X_air, _,_ = real_data_loading(air,24,seed=0)
    air_train, air_test = train_test_split(X_air,test_size=0.2,random_state=0)

    print(f'Train Samples: {air_train.shape}')
    print(f'saved at data/air/train.npy')
    np.save('./data/air/train.npy',air_train)

    print(f'Test Samples: {air_test.shape}')
    print(f'saved at data/air/test.npy')
    np.save('./data/air/test.npy',air_test)


    energy = pd.read_csv("./data/energy_data.csv").values
    X_energy, _,_ = real_data_loading(energy,24,seed=0)
    energy_train, energy_test = train_test_split(X_energy,test_size=0.2,random_state=0)

    print(f'Train Samples: {energy_train.shape}')
    print(f'saved at data/energy/train.npy')
    np.save('./data/energy/train.npy',energy_train)

    print(f'Test Samples: {energy_test.shape}')
    print(f'saved at data/energy/test.npy')
    np.save('./data/energy/test.npy',energy_test)


    ## ECG 5000
    ecg_data = pd.read_fwf("./data/ECG_DATA.txt",header=None)
    y, X = ecg_data.iloc[:,0], ecg_data.iloc[:,1:]
    print(y.value_counts())
    ecg_norm = X[y==1].values

    ecg_norm = ecg_norm[:,:,np.newaxis]

    print(ecg_norm.shape)

    ecg_train, ecg_test = train_test_split(ecg_norm,test_size=0.2,random_state=0)

    print(f'Train Samples: {ecg_train.shape}')
    print(f'saved at data/ecg/train.npy')
    np.save('./data/ecg/train.npy',ecg_train)

    print(f'Test Samples: {ecg_test.shape}')
    print(f'saved at data/ecg/test.npy')
    np.save('./data/ecg/test.npy',ecg_test)

     ## EYE DATASET
    eye_data = pd.read_csv("./data/eeg-eye-state.csv")
    _ , X = eye_data.iloc[:,-1], eye_data.iloc[:,:-1]
    print(X.shape)

    print(X.values)
    eye_data,_,_ = real_data_loading(X.values,128,0)


    eye_train, eye_test = train_test_split(eye_data,test_size=0.2,random_state=0)

    print(f'Train Samples: {eye_train.shape}')
    print(f'saved at data/eye/train.npy')
    np.save('./data/eye/train.npy',eye_train)

    print(f'Test Samples: {eye_test.shape}')
    print(f'saved at data/eye/test.npy')
    np.save('./data/eye/test.npy',eye_test)
    print(eye_test)

    ## STOCK

    stock_data = pd.read_csv("./data/stock_data.csv")
    print(stock_data.shape)

    stock_data,_,_ = real_data_loading(stock_data,24,seed=0)

    stock_train, stock_test = train_test_split(stock_data,test_size=0.2,random_state=0)

    print(f'Train Samples: {stock_train.shape}')
    print(f'saved at data/stocks/train.npy')
    np.save('./data/stocks/train.npy',stock_train)

    print(f'Test Samples: {stock_test.shape}')
    print(f'saved at data/stocks/test.npy')
    np.save('./data/stocks/test.npy',stock_test)


    ### KDD_CUP 

    kdd_cup = np.load("./data/kdd_cup.npy")
    print(kdd_cup.shape)

    

    train, test = train_test_split(kdd_cup,test_size=0.1,random_state=0)

    print(f'Train Samples: {train.shape}')
    print(f'saved at data/kdd_cup/train.npy')
    np.save('./data/kdd_cup/train.npy',train)

    print(f'Test Samples: {test.shape}')
    print(f'saved at data/kdd_cup/test.npy')
    np.save('./data/kdd_cup/test.npy',test)

    ### Trafic hourly
    traffic = np.load("./data/traffic_hourly.npy")
    print(traffic.shape)

    train, test = train_test_split(traffic,test_size=0.1,random_state=0)

    os.makedirs('./data/traffic',exist_ok=True)
    print(f'Train Samples: {train.shape}')
    print(f'saved at data/traffic/train.npy')
    np.save('./data/traffic/train.npy',train)

    print(f'Test Samples: {test.shape}')
    print(f'saved at data/traffic/test.npy')
    np.save('./data/traffic/test.npy',test)






















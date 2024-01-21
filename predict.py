import pandas as pd
import seaborn as sns
import numpy as np
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
from pickle import load

#Add positional command input to get the file name
import sys
file_name = sys.argv[1]

df = pd.read_csv(file_name)

#Load scaler 
with open('preprocess/scaler.pkl', 'rb') as f:
    scaler = load(f)

#Load Onhotencoder
with open('preprocess/onehot_encoder.pkl', 'rb') as f:
    encoder = load(f)

with open('preprocess/pca.pkl', 'rb') as f:
    pca = load(f)

#Load model
model = XGBRegressor(n_estimators=5000, max_depth = 10 , learning_rate=0.05)
model.load_model('models/xgboost_5000_10.json')


X = df.copy()

X['Slickwater'] = 0
X['Crosslink'] = 0
X['Linear'] = 0

X['Slickwater'] |= X['ffs_frac_type'].str.contains('Slickwater')
X['Crosslink'] |= X['ffs_frac_type'].str.contains('Crosslink')
X['Linear'] |= X['ffs_frac_type'].str.contains('Linear')

X.drop('ffs_frac_type', axis=1, inplace=True)

X['Slickwater'] = X['Slickwater'].astype(np.int8)
X['Crosslink'] = X['Crosslink'].astype(np.int8)
X['Linear'] = X['Linear'].astype(np.int8)

X = X.drop('frac_type', axis = 1)
X.dropna(axis = 0, inplace = True)

def encode_categorical_columns(df):
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns
    encoded_data = encoder.transform(X[categorical_columns])
    encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_columns))
    return pd.concat([df.drop(categorical_columns, axis=1), encoded_df], axis=1)

X_encoded = encode_categorical_columns(X)

unknown_cols = [c for c in X_encoded.columns if 'Unknown' in c]
X_encoded.drop(unknown_cols, axis =1, inplace=True)

X_encoded.replace([np.inf, -np.inf], np.nan, inplace = True)
X_encoded.dropna(inplace = True)

#Selecting features 
selected_features = ['surface_x', 'surface_y', 'gross_perforated_length', 'total_proppant', 'true_vertical_depth', 'proppant_intensity', 'frac_fluid_intensity', 'average_stage_length', 'average_proppant_per_stage', 'average_frac_fluid_per_stage', 'frac_fluid_to_proppant_ratio', 'bin_lateral_length', 'horizontal_midpoint_x', 'horizontal_midpoint_y', 'horizontal_toe_x', 'horizontal_toe_y', 'Slickwater', 'Crosslink', 'Linear', 'relative_well_position_Inner Well', 'batch_frac_classification_Non-Batch Frac']
# Load the data
X_encoded = X_encoded[selected_features]


#Scale the data
X_scaled = scaler.fit_transform(X_encoded) 

prediction = model.predict(X_scaled)

prediction_df = pd.DataFrame(data=prediction, columns=['OilPeakRate'])

prediction_df.to_csv('submit.csv', index=False)



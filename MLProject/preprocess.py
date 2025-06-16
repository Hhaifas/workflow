from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from joblib import dump
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split
import pandas as pd


def preprocess_data(data, target_column, save_path, file_path):
    # Menentukan fitur numerik dan kategoris
    numeric_features = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
    categorical_features = data.select_dtypes(include=['object']).columns.tolist()
    
    column_names = data.columns
    column_names = data.columns.drop(target_column)
    # Membuat DataFrame kosong dengan nama kolom
    df_header = pd.DataFrame(columns=column_names)
    
    # Menyimpan nama kolom sebgai header tanpa data
    df_header.to_csv(file_path, index=False)
    print(f"Nama kolom berhasil disimpan ke: {file_path}")
    
    # Pastikan target_column tidak ada di numeric_features atau categorical features
    if target_column in numeric_features:
        numeric_features.remove(target_column)
    if target_column in categorical_features:
        categorical_features.remove(target_column)
        
    # Pipeline untuk fitur numerik
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    
    # Pipeline untuk fitur kategoris
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encoder', OrdinalEncoder())
    ])    
    
    
    # Column Transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    # Memisahkan target
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    # Membagi data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Fitting dan transformasi data pada training set
    X_train = preprocessor.fit_transform(X_train)
    # Transformasi pada data testing
    X_test = preprocessor.transform(X_test)
    
    # Simpan pipeline
    dump(preprocessor, save_path)
    
    return X_train, X_test, y_train, y_test
    
    # Encoder untuk target
    # target_encoder = LabelEncoder()
    # y_train = target_encoder.fit_transform(y_train)
    # y_test = target_encoder.transform(y_test)
    # dump(target_encoder, target_encoder_path)
    
    # if hasattr(preprocessor, "get_feature_names_out"):
    #     feature_names = preprocessor.get_feature_names_out()
    # else:
    #     feature_names = numeric_features + categorical_features
        
    # # dataframe fitur hasil transformasi
    # X_train = pd.DataFrame(X_train, columns=feature_names)
    
    # X_train[target_column] = y_train
    # X_train.to_csv(file_path, index=False)

    
 

# data = pd.read_csv("preprocessing/personality_dataset_preprocessing.csv")
# X_train, X_test, y_train, y_test = preprocess_data(data, 
#                                                    'Personality', 
#                                                    'preprocessor_pipeline.joblib', 
#                                                    'data.csv',
#                                                    'target_encoder.joblib')

# Validasi memastikan tahapan preprocessing dilakukan dan disimpan dengan baik
# def inference(new_data, load_path):
#     # Memuat pipeline preprocessing
#     preprocessor = load(load_path)
#     print(f"Pipeline preprocessing dimuat dari: {load_path}")
    
#     # Transformasi data baru
#     transformed_data = preprocessor.transform(new_data)
#     return transformed_data


# def inverse_transform_data(transformed_data, load_path, new_data_columns, num_cols, cat_cols):
#     preprocessor = load(load_path)
#     numeric_transformer = preprocessor.named_transformers_['num']['scaler']
#     categorical_transform = preprocessor.named_transformers_['cat']['encoder']
#     n_num = len(num_cols)
#     n_cat = len(cat_cols)

#     # Slicing sesuai urutan pipeline (asumsikan num_cols dulu, lalu cat_cols)
#     transformed_numeric_data = transformed_data[:, :n_num]
#     transformed_categorical_data = transformed_data[:, n_num:n_num+n_cat]

#     original_numeric_data = numeric_transformer.inverse_transform(transformed_numeric_data)
#     original_categorical_data = categorical_transform.inverse_transform(transformed_categorical_data)

#     # Buat dataframe kosong dengan urutan kolom asli
#     inversed_data = pd.DataFrame(index=range(transformed_data.shape[0]), columns=new_data_columns)

#     # Isi kolom satu per satu sesuai urutan aslinya
#     num_idx = 0
#     cat_idx = 0
#     for col in new_data_columns:
#         if col in num_cols:
#             inversed_data[col] = original_numeric_data[:, num_idx]
#             num_idx += 1
#         elif col in cat_cols:
#             inversed_data[col] = original_categorical_data[:, cat_idx]
#             cat_idx += 1

#     return inversed_data


# import numpy as np
# pipeline_path = 'preprocessor_pipeline.joblib'
# col = pd.read_csv("data.csv")
# df = pd.read_csv("preprocessing/personality_dataset_preprocessing.csv")
# df = df.drop(columns=['Personality'])

# # new_data = [4.0, 6.0, 2.0, 0.0, 8.0, "No", "No"]

# # Mengubah menjadi numpy.ndarray
# # new_data = np.array(df)

# # new_data = pd.DataFrame([new_data], columns=col.columns)
# transformed_data = inference(df, pipeline_path)

# # num_cols = ['Time_spent_Alone',	'Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Post_frequency']
# # cat_cols = ['Stage_fear', 'Drained_after_socializing']

# columns = ['Time_spent_Alone',	'Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Post_frequency',
#            'Stage_fear', 'Drained_after_socializing']

# # inversed_data = inverse_transform_data(transformed_data, pipeline_path, new_data.columns, num_cols, cat_cols)
# # transformed_feature_names = pipeline_path.get_feature_names_out()

# transformed_df = pd.DataFrame(transformed_data, columns=columns)
# transformed_df.to_csv("data.csv", index=False)

# # Output hasil preprocessing dan inference
# # print("Data setelah preprocessing (training):")
# # print(df)
# # print("\nData baru setelah transformasi:")
# # print(transformed_data)
# # print("\nData setelah inverse transform:")
# # print(inversed_data)
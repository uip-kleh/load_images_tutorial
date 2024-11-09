import pandas as pd
from keras_preprocessing.image import ImageDataGenerator 

if __name__ == "__main__":
    df = pd.read_csv("data_info.csv")
    print(df.columns)
    print(df["path"].head())
    image_gen = ImageDataGenerator()
    image_data = image_gen.flow_from_dataframe(
        dataframe=df, 
        directory="images",
        x_col="path",
        y_col="label"
    )
    
    print(image_data[0][0].shape)
    print(image_data.labels)
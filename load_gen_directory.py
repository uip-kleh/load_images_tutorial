from keras_preprocessing.image import ImageDataGenerator 

if __name__ == "__main__":
    image_gen = ImageDataGenerator()
    image_data = image_gen.flow_from_directory(
        directory="images"
    )
    
    print(image_data[0][0].shape)
    print(image_data.labels)
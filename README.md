# Machine Learning NutriCheck

## Introduction

In our project, NutriCheck, we performed image classification using CNN algorithm and transfer learning with MobileNetV2. We added several layers to adapt the model to our dataset, then fine-tuned it by unfreezing the top 20 layers to improve accuracy. The model achieved 85% accuracy on training and 82% accuracy on validation. Then each label in the dataset has its nutrition data stored in CSV format.

## Model Architecture
The model used in the NutriCheck project is based on MobileNetV2 with transfer learning for image classification. Below is the detailed explanation of the architecture:

1. Pre-trained MobileNetV2 Backbone:
The model uses MobileNetV2 as the backbone, initialized with ImageNet weights. The top classification layers are excluded (include_top=False), and global average pooling (pooling='avg') is applied to reduce the output to a fixed size, making it more suitable for the custom classification task.

2. Freezing the Pre-trained Layers:
Initially, all layers of the pre-trained MobileNetV2 model are frozen (layer.trainable = False) to preserve the learned features from ImageNet. This allows the model to focus on learning features specific to the new dataset without altering the pre-trained weights.

3. Custom Dense Layers:
Two custom fully connected (dense) layers are added after the MobileNetV2 backbone. Both layers have 128 units and use the ReLU activation function to help the model learn more complex patterns in the dataset before making the final classification.

4. Output Layer:
The final layer of the model is a dense layer with 53 units and a softmax activation function, which is used for multi-class classification. The 53 units correspond to the number of classes in the dataset, allowing the model to output class probabilities.

5. Model Compilation:
The model is compiled using the Adam optimizer with a learning rate of 1e-4 and categorical cross-entropy as the loss function. This setup is ideal for multi-class classification tasks and ensures efficient optimization during training.

6. Callbacks:
During training, several callbacks are used to enhance the model's performance. The ModelCheckpoint callback saves the best model based on validation accuracy, ReduceLROnPlateau reduces the learning rate when validation loss plateaus, and a custom callback stops training early if the training accuracy exceeds 85% and validation accuracy exceeds 82%.

7. Fine-tuning:
After the initial training phase, the top 20 layers of the pre-trained MobileNetV2 model are unfrozen (for layer in pre_trained_model.layers[-20:]) to fine-tune the model. Fine-tuning allows the model to adjust the pre-trained layers to better fit the new dataset, with a lower learning rate of 1e-5 to prevent overfitting.

8. Training:
The model is trained for 20 epochs initially, followed by 100 epochs of fine-tuning. Training is performed using data generators for both the training and validation sets, and the process is optimized with the help of the callbacks.

## Dataset
Our dataset consists of two types of data: image data for food classification and nutritional data stored in a CSV file, which contains detailed nutritional information for each food item
### Food Image
For the food images in our project, we collected data from the internet using an API_KEY and CSE_ID, successfully gathering images for 10 Indonesian food classes. These classes include: 
* Yellow Rice
* Tofu and Rice Cake
* Grilled Chicken
* Chicken Porridge
* Green Bean Porridge
* Grilled Fish
* Chicken Noodles
* Uduk Rice
* Oxtail Soup
* Chicken Soup. 

Additionally, for other food images, we sourced them from Kaggle's [Food Images (Food-101)](https://www.kaggle.com/datasets/kmader/food41) dataset and the [Indonesian Food](https://www.kaggle.com/datasets/theresalusiana/indonesian-food) dataset. Afterward, we refined the class names to align with foods commonly consumed in Indonesia. This process involved selecting food categories that reflect popular Indonesian cuisine. In total, we used 53,000 images, which were divided into 53 classes. The dataset was then split into training, validation, and test sets with a distribution of 80% for training, 10% for validation, and 10% for testing. As a result, the training set contains 800 images per class, the validation set contains 1000 images, and the test set contains 100 images per class.

![Distribusi Data Pelatihan](/assets/Training%20Distribution.png)
![Distribusi Data Validasi](/assets/Validation%20Distribution.png)


### Nutrition CSV
The text data, which contains nutritional information, is stored in a CSV file and was obtained from the FatSecret API platform. Access to the API is granted using a consumer key and consumer secret, which are provided after registering on the FatSecret platform. To retrieve the nutritional data, we used Google Colab as the environment and Python as the programming language. First, we entered the food name in the food search, obtained its corresponding ID, and then used this ID to fetch the nutritional data, which was subsequently saved in a file named 'nutrition.csv'. The data then went through a preprocessing stage, and the final processed data was saved in 'clean_data.csv'. The features contained in the cleaned dataset include:

| Feature              | Description                                                                                      |
|----------------------|--------------------------------------------------------------------------------------------------|
| *Food ID*          | A unique identifier assigned to each food item.                                                   |
| *Food Name*        | The name of the food item.                                                                        |
| *Calories*         | The amount of energy provided by the food, measured in kilocalories (kcal).                      |
| *Total Carbohydrate* | The total amount of carbohydrates in the food, including sugars, starches, and fiber, measured in grams. |
| *Protein*          | The amount of protein in the food, measured in grams.                                             |
| *Dietary Fiber*    | The amount of fiber present in the food, important for digestive health
| *Vitamin A*        | The amount of vitamin A, important for vision and immune function
| *Vitamin C*        | The amount of vitamin C, important for immune health and skin repair
| *Vitamin B*        | The amount of vitamin B complex, supports energy production and nerve function |
| *Iron*             | The amount of iron, essential for oxygen transport in the blood
| *Calcium*          | The amount of calcium, vital for bone health.                       |
| *Serving Size*     | The recommended portion size for the food item.                                                   |
| *Serving Size (grams)* | The weight of the serving size, measured in grams.                                              |

## Saved Model
The trained model is saved in the .h5 format, which consolidates the model's architecture, weights, and optimizer information into a single file. This format is commonly used for deployment on servers and integrates seamlessly with frameworks like Flask, enabling the creation of APIs to make the model accessible for various applications.

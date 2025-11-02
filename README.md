# second-hand-car-price-prediction
A deep learning project that builds and trains a neural network model to predict the prices of second-hand cars based on various features. The model is trained using preprocessed data and evaluated using metrics like Mean Absolute Error (MAE) and Loss.
# Car Price Prediction Using Neural Networks

This project involves building and training a neural network model to predict car prices based on various features. The dataset is preprocessed, and several neural network models with different configurations are evaluated to achieve the lowest Mean Absolute Error (MAE).

## Project Description

In this project, the following steps are applied:

1. **Importing Libraries**:
   - Essential libraries for data processing, model training, and visualization are imported:
     - **Pandas** for data manipulation
     - **NumPy** for mathematical operations
     - **Scikit-learn** for preprocessing and modeling
     - **Matplotlib** and **Seaborn** for plotting graphs

2. **Data Preprocessing**:
   - Missing values are handled and categorical features are encoded using **OneHotEncoder**.
   - Numerical features are normalized to ensure the model trains effectively.
   - The dataset is then split into training, validation, and test sets using **train_test_split**.

3. **Neural Network Model**:
   - A neural network model is built with three hidden layers to predict car prices based on their features.
   - The model is trained, and during the training process, **Loss** and **Mean Absolute Error (MAE)** metrics are displayed.

4. **Model Evaluation**:
   - The trained model is evaluated on the test set, and predictions are compared with actual values.
   - Graphs are plotted to visualize the training process and the comparison between predicted and actual values.

5. **Model Comparison**:
   - Models with different numbers of neurons and hidden layers are trained.
   - The model with the lowest MAE is saved and used for comparison.
   - A comparative graph is plotted to show the performance of different models.

6. **Conclusion**:
   - This project demonstrates the process of building and training a neural network model for car price prediction, including data preprocessing, model evaluation, and comparison of different configurations.

## Libraries Used
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- TensorFlow (Keras)

## How to Use
1. **Clone the repository**:
2. **Install required libraries**:
3. **Run the code**:
Open the `deep_learning_s2p3.ipynb` file in Jupyter Notebook and follow the steps for data preprocessing, training, and evaluating the neural network model.

## Contributing
If you would like to contribute to this project, please feel free to submit a pull request. All suggestions and improvements are welcome!

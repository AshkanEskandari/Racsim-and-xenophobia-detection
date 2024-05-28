# Detecting Racism and Xenophobia in Text

## Introduction
This project aims to develop a deep learning model to detect racism and xenophobia in text data. The process involves gathering datasets, preprocessing text, labeling data, and training the model. This documentation outlines each step in detail, providing an overview of the methodologies used and the outcomes achieved.

## Data Collection

### Datasets Used

1. **Ethos Hate Speech Dataset**:
   - URL: [Ethos Dataset](https://raw.githubusercontent.com/intelligence-csd-auth-gr/Ethos-Hate-Speech-Dataset/master/ethos/ethos_dataEthos_Dataset_Multi_Label.csv)
   - Description: Contains labels for racism and xenophobia.

2. **HateEvalTeam Dataset**:
   - URL: [HateEvalTeam Dataset](https://raw.githubusercontent.com/cicl2018/HateEvalTeam/master/Data%20Files/Data%20Files/%232%20Development-English-A/train_dev_en_merged.tsv)
   - Description: A large dataset for hate speech detection without specific labels for racism and xenophobia.

3. **NCBI Hate Speech Dataset**:
   - URL: [NCBI Dataset](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9044360/)
   - Description: This dataset was in Spanish and was translated to English using the Google API. It did not contain specific labels for racism and xenophobia.

### Translation
The NCBI dataset, originally in Spanish, was translated to English using the Google Translation API to ensure consistency in language across all datasets.

## Data Labeling
The Ethos dataset provided explicit labels for racism and xenophobia. However, the other datasets did not. The approach taken was:

1. **Using Ethos Labels**: The Ethos dataset’s labels were used to train an initial model.
2. **Self-Training for Label Propagation**: The trained model was then used to predict and label the unlabeled datasets.

### Combining Datasets
All three datasets were combined into a single dataset. The text data was vectorized using TF-IDF (Term Frequency-Inverse Document Frequency) with a maximum of 1000 features. This transformed text data was then used for training.

### Setting Maximum Length
The maximum length for text sequences was set to cover 95% of the text lengths in the dataset, ensuring most texts are included without excessive padding.

## Model Training

### Initial Training

1. **Data Splitting**: The labeled data was split into training and validation sets.
2. **Model Selection**: A RandomForestRegressor wrapped in a MultiOutputRegressor was chosen as the initial model to predict both racism and xenophobia labels.
3. **Training**: The initial model was trained on the labeled data and evaluated using Mean Squared Error (MSE) on the validation set.

### Self-Training for Unlabeled Data
A self-training function was implemented to iteratively predict labels for the unlabeled data with high confidence. This process involved:

1. Predicting the labels for the unlabeled data.
2. Selecting the predictions with high confidence (above a defined threshold).
3. Adding these high-confidence predictions to the labeled dataset.
4. Retraining the model on the expanded labeled dataset.

This process was repeated until no high-confidence predictions could be made.


### Embedding Layer
GloVe embeddings were used for text representation in the final deep learning model. Various neural network structures were experimented with to optimize performance.

### Potential Use of Pre-trained Models
To further enhance the model, pre-trained models like BERT (uncased) can be used. These models, pre-trained on large corpora, can offer better performance by leveraging transfer learning and rich contextual embeddings.


### Observations
- The datasets had an insufficient amount of text about specific groups, such as Muslims, which could affect model performance.
- The iterative self-training approach improved the coverage of the labels across the combined dataset.

### Future Work
- Improve dataset diversity by incorporating more text samples related to various groups.
- Experiment with more advanced deep learning architectures and hyperparameter tuning.
- Explore transfer learning techniques by using pre-trained models like BERT to enhance model accuracy.

## Conclusion
This project successfully developed a model to detect racism and xenophobia in text by combining multiple datasets and employing a self-training methodology for labeling. While the model shows promising results, further improvements can be made by addressing dataset limitations and refining the model architecture.

## References

- Ethos Hate Speech Dataset: [Ethos Dataset](https://raw.githubusercontent.com/intelligence-csd-auth-gr/Ethos-Hate-Speech-Dataset/master/ethos/ethos_dataEthos_Dataset_Multi_Label.csv)
- HateEvalTeam Dataset: [HateEvalTeam Dataset](https://raw.githubusercontent.com/cicl2018/HateEvalTeam/master/Data%20Files/Data%20Files/%232%20Development-English-A/train_dev_en_merged.tsv)
- NCBI Hate Speech Dataset: [NCBI Dataset](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9044360/)

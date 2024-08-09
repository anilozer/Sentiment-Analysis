# Sentiment Analysis on Amazon Kindle Book Reviews
Overview
This notebook demonstrates a complete end-to-end sentiment analysis pipeline applied to Amazon Kindle Book Reviews. The analysis leverages a deep learning technique called Long Short-Term Memory (LSTM) network to classify reviews as either positive or negative based on their text content.

Dataset
The dataset used for this project is the Amazon Kindle Book Review for Sentiment Analysis available on Kaggle. I opted to use the unprocessed version of the dataset and handled all preprocessing tasks within this notebook.

Preprocessing Steps
Column Selection:

Selected relevant columns: rating and reviewText.
Simplified the rating column to binary classification: Ratings 1, 2, and 3 were labeled as negative (0), while ratings 4 and 5 were labeled as positive (1).

Data Cleaning:

Checked for null values and confirmed none were present in the relevant columns.
Converted all review texts to lowercase for uniformity.

Tokenization:

Tokenized the text data, converting words into sequences of integers.
Set a vocabulary limit of the 20,000 most frequent words and padded sequences to ensure uniform length.

Data Splitting:

Split the dataset into training (60%), validation (20%), and testing (20%) sets.
Word Embeddings
For word embeddings, I used the pre-trained GloVe vectors from the Wikipedia 2014 and Gigaword 5 dataset. This step involved:

Loading the GloVe vectors and creating an embedding matrix to map words to their corresponding vectors.
Using the embedding matrix as the weights for the embedding layer in our neural network, ensuring the embedding layer remains non-trainable.
Model Development
I experimented with various LSTM models to determine the optimal architecture for this sentiment analysis task:

Initial Model:

Architecture: 64 LSTM units, 10 epochs, batch size of 64.
Performance: Achieved 78% accuracy on the test set, but with a high loss value indicating room for improvement.
Model Tuning:

Reduced LSTM units to 8 and epochs to 5.
Result: Lower accuracy (75.8%) and higher loss, indicating a less effective model.
Final Model:

Increased LSTM units to 128 and epochs to 15, with a larger batch size of 128.
Result: Improved performance, achieving ~78% accuracy on the test set, with better loss values compared to previous models.

Conclusions
The final model, with 128 LSTM units, provided the best balance between accuracy and loss, achieving a test accuracy of approximately 78%.
The results indicate that while the model performs reasonably well, there is potential for further refinement, such as hyperparameter tuning or experimenting with different model architectures like bidirectional LSTMs or GRUs.

How to Use This Notebook
To reproduce the results, follow these steps:

Download the dataset from the provided Kaggle link.
Ensure you have the necessary libraries installed (pandas, numpy, tensorflow).
Run the notebook cells sequentially, starting with data preprocessing, through to model training, and evaluation.
Modify and experiment with the model parameters to see how changes affect the performance.

Future Work
Potential improvements could involve:

Exploring different neural network architectures.
Applying more sophisticated text preprocessing techniques.
Using more advanced word embeddings.








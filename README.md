# DESCRIPTION

This repository contains code for cleaning & processing data, creating an LSTM-based model to extract skills from text data, and generating job fit scores for resumes. The model is implemented using TensorFlow and Keras and utilizes pre-trained GloVe word embeddings for enhanced performance.


# INSTALLATION

If only need to view the final dashboards, download **Visualization** -> **Dashboard.twbx**.

Otherwise:

Download the GloVe word embeddings file (glove.6B.100d.txt) and place it in the project directory.

Download the job data, resume data & .shp files.

Open the Jupyter Notebooks using a platform like Google Colab or Jupyter that links to the project directory.
## Folder Description

### Datasets:
- **Data Souce Link.txt**: [Link to raw scraped data with 160K+ job postings from Glassdoor (Kaggle)](https://www.kaggle.com/datasets/andresionek/data-jobs-listings-glassdoor?select=glassdoor.csv).
- **resume_dataset.csv**: Sample data used to check model performance in job recommendation.
- **ne_50m_admin_0_countries.shp**: Shapefile used for data visualization and extracting missing coordinates.
- **glove.6B.100d.txt**: [Pre-trained GloVe word embeddings file reference.](https://medium.com/@Olohireme/job-skills-extraction-from-data-science-job-posts-38fd58b94675

### Data Preparation:
- **translate_desc_to_en.ipynb**:(my contrinution to coding -100%) Jupyter notebook containing code for cleaning the raw data. Steps include text translation, fuzzy job title matching, generating coordinates from location name etc.
- **Reverse Geocoding.ipynb**:(contribution-none) Fill missing Country using coordinates.
- **data_prep_for_final_model.ipynb**:(contribution-100%) Jupyter notebook - preparing the data for the model. includes grouping sectors, removing clusters with small sample sizes etc.

### Model building and validation:
- **LSTM_model_creation.ipynb**: [Jupyter Notebook containing the code for LSTM model creation.Reference](https://medium.com/@Olohireme/job-skills-extraction-from-data-science-job-posts-38fd58b94675)
- **np_train_skills_no_commas.csv**: CSV file containing the training data.
- **lstm_skill_extractor.h5**: Saved model file.
- **keyword_extraction_model-main.ipynb**:(contribution-100%) Code for extracting keywords using the model output and further data processing and cleaning and data manipulation. Includes job similarity score calculation between different job titles.
- **Job_fit_scoring_based_on_resume_input.ipynb**: Code for generating job fit scores for resumes.

### Visualization:
- **Dashboard.twbx**: Final Tableau Dashboard with 2 views - Map Viz and Job Recommendation.


# EXECUTION

Ensure you have the necessary dependencies installed:

```
pip install pandas tensorflow keras tqdm nltk numpy matplotlib seaborn scikit-learn googletrans beautifulsoup4 geopandas
```


Execute the cells in the notebooks sequentially to clean the data, train the model and generate results.

The training process includes loading the dataset, tokenizing text, creating word embeddings, and training the LSTM model.

The trained model is saved as **lstm_skill_extractor.h5** in the same project directory.

The model's performance is evaluated using accuracy, precision, recall, and a confusion matrix on a test set.

The model output is used to predict the best job fit for a sample of resume and compared the recommendation with actual job the resume was applied for.

### Additional Notes
The code assumes a binary classification task (skill extraction), and the model architecture can be adjusted for different tasks or datasets.
Adjust hyperparameters, such as the learning rate, batch size, and dropout rates, as needed.
Ensure the dataset (np_train_skills_no_commas.csv) adheres to the required format.

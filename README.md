# Disaster Response Pipeline Project
This is a project completed as part of the Udacity Data Science Nanodegree.
### Purpose:
The purpose of this project is to analyse social media messages that are sent during times of natural disasters. There are lots of messages posted during these times and it is critical to try and identify the legitimate messages that are reporting topics that are of use to the emergency services such as where aid is required or where the natural disaster is getting worse.

Data from FigureEight contains labelled tweets that allow us to create a machine learning model that can identify topics from future messages.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
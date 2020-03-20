# Disaster Response Pipelines
This is a project completed as part of the Udacity Data Science Nanodegree.
##### Purpose
The purpose of this project is to analyse social media messages that are sent during times of natural disasters. There are lots of messages posted during these times and it is critical to try and identify the legitimate messages that are reporting topics that are of use to the emergency services such as where aid is required or where the natural disaster is getting worse.

Data from FigureEight contains labelled tweets that allow us to create a machine learning model that can identify topics from future messages.

##### Software used
Python 3.8.1

##### Libraries used 
- json
- plotly
- pandas
- nltk
- flask 
- plotly
- pickle
- sqlalchemy
- re
- sklearn
- numpy

##### Files included in repository and how to run them
- ETL Pipeline Preparation.ipynb - Notebook containing the ETL process that was designed to clean the data into a usable format. Running all cells within the notebook will create the correct SQLite database.
- ML Pipeline Preparation.ipynb - Notebook containing the machine learning scripts that analyse the data and create the model. Running all cells within the notebook will create a model file to be used by the Flask app.
- data/messages.csv - Data containing the content of the message that was written.
- data/categories.csv - Data that contains what category each message is assigned to.
- data/DisasterResponse.db - Sqlite database that contains the data from the two previously mentioned files after cleaning and merging has been done.
- data/process_data.py - Python file that contains code from 'ETL Pipeline Preparation.ipynb', can be run to quickly re-create DisasterResponse.db without running the entire ETL notebook. This can be run by using the following command in a terminal:
```
 python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db
```
- models/train_classifier.py - Python file that contains code from 'ML Pipeline Preparation.ipynb', can be run to quickly create a model file called clasifier.pkl. This can be run by using the following command in a terminal:
```
python train_classifier.py ../data/DisasterResponse.db classifier.pkl
```
- models/classifier.pkl - Pickle file that contains the model created by train_classifier.py
- models/modelv1.pkl - Pickle file that contains the model created by 'ML Pipeline Preparation.ipynb' (Not used anywhere, just included for completion).
- app/run.py - Flask app that visualises some aspects of the data and allows users to see what the recommendation is for a particular message. This can be run by using the following command in a terminal and then navigating to 127.0.0.1:3001
```
 python run.py
```
- app/templates - Folder that contains two html files used by the flask app.
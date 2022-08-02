# Disaster Response Pipeline and Web App project

## Motive

For this project, we created a website that analyzes disaster text messages 
to understand the situation the sender is in.
By entering a message through the created web app, you can figure out what kind of emergency situation you are in. 
The provided data includes the situation and message of the sender of the message, 
and the provided data has been processed for machine learning. 
Also, to get the best results, I applied various parameters using gridsearch from Scikit learn.

Disaster text data were provided by Figure Eight.


## Files
### app    

| - template    
| |- master.html # main page of web app    
| |- go.html # classification result page of web app    
|- run.py # Flask file that runs app    


### data    

|- disaster_categories.csv # data to process    
|- disaster_messages.csv # data to process    
|- process_data.py # data cleaning pipeline    
|- InsertDatabaseName.db # database to save clean data to     


### models   

|- train_classifier.py # machine learning pipeline     
|- classifier.pkl # saved model     


README.md    


## How to run web app
1. Run `python run.py` in terminal
2. Go to http://0.0.0.0:3001 in your browser

![screenshot1](./imgs/~)

![screenshot1](./imgs/~)

![screenshot1](./imgs/~)


## License
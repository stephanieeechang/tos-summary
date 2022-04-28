# Terms of Service Agreement and Privacy Policy Analysis
Perform extractive summarization on Terms of Service and Privacy Policy documents using model derived from BERT.

## Installation
`pip install requirements.txt`

## Execution
To test extractive summarization model on the Plain English Summarization of Contracts dataset, run this in `src` directory: 
```
python main.py
```
To test extractive summarization model on custom input text, run this in `src` directory: 
```
python main.py --demo_mode
```

For other command line arguments available, see `main.py`.

## Directory Overview
`data`: Contains datasets used to train and test extractive summarization model.\
`src`: Contains code for BERT-based extractive summarization model.\
`Twitter`: Contains code for Twitter topic modeling.\
`utils`: Contains code for Reddit topic modeling.

import json

with open('pubilic-sentiment_NPL.ipynb.json') as data:
    jsonToPython = json.loads(data)
    jsonToPython.close()
    print(jsonToPython)

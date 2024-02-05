import pandas as pd
import numpy as np

df = pd.read_json('foundLinks.json')
links = set()


# Checks if it is a valid link, removes emails, phone numbers, and irrelevant links ('i.e. linkedin, youtube, facebook, etc.)
def isValidURL(url):

    validLinks = [] # can specify ombuds, equityandinclusion, etc. THE SCOPE
    invalidLinks = ['facebook', 'twitter', 'linkedin', 'youtube', 'instagram']

    # Check if beginning is valid and in emory domain
    if(url[:4]!='http' or 'emory' not in url):
        return False


    # Remove useless links
    for subLink in invalidLinks:
        if(subLink in url):
            return False
    
    # # Remove links that don't have necessary elements (i.e. emory, ombuds if you want to be more specific)
    # for subLink in validLinks:
    #     if(subLink in url):
    #         return True

    # Return false if sublink doesn't match link
    return True


n, d = df.shape

# Iterate through all base url's and add to the set if they are valid
for i in range(n):
    url = df.iloc[i][0]
    if(isValidURL(url)):
        links.add(url)

for i in range(n):
    urls = df.iloc[i][1]
    for url in urls:
        if(isValidURL(url)):
            links.add(url)

# Convert set to a df
finalList = list(links)
# Create a Pandas DataFrame from the list
finalDF = pd.DataFrame(finalList, columns=["links"])

# Write csv files
finalDF.to_csv('emoryLinks.csv',index=False)


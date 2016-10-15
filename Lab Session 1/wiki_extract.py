'''
Created on 21-May-2015

@author: Anantharaman Narayana Iyer
'''
import os
import  wikipedia 

ds_path = os.path.join("datasets")
def remove_non_ascii(text):
    return ''.join([i if ord(i) < 128 else ' ' for i in text])

def get_pages(names): # names is a list of names to be queried on wiki
    pages = []
    for name in names:
        page = wikipedia.page(name, auto_suggest = True)
        if page != None:
            pages.append({"name": name, "page": page})        
    return pages

def create_ds(names, pth): # names is a list of names to be queried on wiki
    pages = get_pages(names)
    for page in pages:
        text = page["page"].content
        if not os.path.exists(page["name"]):
            os.makedirs(page["name"])
        f = open(os.path.join(pth, page["name"]+".txt"), "wb")
        f.write(remove_non_ascii(text))
        f.close()
    return pages

if __name__ == '__main__':
    names = [
             # "gandhi", "nehru", "sachin tendulkar", "katrina kaif",
             # "Albert Einstein", "John F Kennedy", "Narendra Modi",
             # "Rajinikanth", "Visvesvaraya", "hema malini", 
             # "amitabh bachchan", "indira gandhi",
             # "Ashoka", "Don Bradman", "Abdul Kalam", "Pele",
             # "Rajiv Gandhi", "Thomas Alva Edison", "Subramania Bharathiyar", ""
			 # "Maharana Pratap", "Hrithik Roshan"
			 "sachin tendulkar", "rahul dravid"
            ]
    pages = create_ds(names, ds_path)

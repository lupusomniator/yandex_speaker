import pickle
df = 'sdf'
with open("canonized",'wb') as file:
    pickle.dump(df, file)
    
    
with open("canonized",'rb') as file:
    ass = pickle.load(file)
    
print(ass)

# Randomising data function 

def randomise(data):
    data = data.iloc[:, 0:17].sample(frac=1).reset_index(drop=True)
    return data
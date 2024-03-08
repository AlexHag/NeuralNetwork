import pickle

def num_to_char(num):
    if 0 <= num <= 25:
        return chr(num + 65)
    elif 26 <= num <= 35:
        return str(num - 26)
    else:
        raise ValueError("Number out of range. Must be between 0 and 35.")

def save_model(model, filename):
    for i in range(len(model)):
        try:
            model[i].pop('layer')
        except:
            pass
        try:
            model[i].pop('d_layer')
        except:
            pass
        try:
            model[i].pop('layer_activation')
        except:
            pass
        try:
            model[i].pop('d_weight')
        except:
            pass
        try:
            model[i].pop('d_bias')
        except:
            pass
    with open(filename, "wb") as f:
        pickle.dump(model, f)
    
def load_model(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)
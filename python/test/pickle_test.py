import pickle

pickle.dump(favorite_color, open("save.p", "wb"))

favorite_color = pickle.load(open("save.p", "rb"))

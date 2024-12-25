import pickle

# File path of the pickle file
file_path = "temp.pkl"

# Load the object from the pickle file
with open(file_path, "rb") as file:  # "rb" is for reading binary files
    loaded_object = pickle.load(file)
    print(loaded_object)
    print([box.tolist() for box in loaded_object.pred_boxes])
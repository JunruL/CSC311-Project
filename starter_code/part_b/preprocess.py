from scipy.sparse import load_npz
import pandas as pd
import numpy as np
import csv
import os
import ast


def load_student_meta(root_dir="../data"):
    path = os.path.join(root_dir, "student_meta.csv")
    if not os.path.exists(path):
        raise Exception("The specified path {} does not exist.".format(path))
    # Initialize the data.
    data = {
        "user_id": [],
        "gender": [],
        "date_of_birth": [],
        "premium_pupil": []
    }
    # Iterate over the row to fill in the data.
    with open(path, "r") as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            try:
                data["user_id"].append(int(row[0]))
                data["gender"].append(int(row[1]))
                data["date_of_birth"].append(row[2])
                data["premium_pupil"].append(int(row[3]))
            except ValueError:
                # Pass first row.
                pass
            except IndexError:
                # is_correct might not be available.
                pass

    return data


def get_student_meta():
    student_data_raw = load_student_meta()
    student_data_processed = {}

    # process age: from date of birth
    for i in range(len(student_data_raw["user_id"])):
        user_id = student_data_raw["user_id"][i]
        gender = student_data_raw["gender"][i]
        gender /= 2.0
        premium_pupil = student_data_raw["premium_pupil"]
        year = student_data_raw["date_of_birth"][i][0:4]
        if year != '':
            age = 2022 - int(year)
        else:
            age = 15
        age /= 20.0

        student_data_processed[user_id] = (gender, age, premium_pupil)

    return student_data_processed


def get_student_vectors():
    student_data_raw = load_student_meta()
    gender_vector = np.zeros((542, 1))
    age_vector = np.zeros((542, 1))
    for i in range(542):
        gender = student_data_raw["gender"][i]
        gender /= 2.0
        premium_pupil = student_data_raw["premium_pupil"]
        year = student_data_raw["date_of_birth"][i][0:4]
        if year != '':
            age = 2022 - int(year)
        else:
            age = 15
        age /= 20.0
        
        gender_vector[i, 0] = gender
        age_vector[i, 0] = age
    
    return gender_vector, age_vector


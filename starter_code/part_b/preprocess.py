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
    gender_vector = np.zeros((388, 1))
    age_vector = np.zeros((388, 1))
    for i in range(388):
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
    

def load_subject_meta(root_dir="../data"):
    path = os.path.join(root_dir, "subject_meta.csv")


def load_question_meta(root_dir="../data"):
    path = os.path.join(root_dir, "question_meta.csv")
    if not os.path.exists(path):
        raise Exception("The specified path {} does not exist.".format(path))

    df = pd.read_csv(path)
    df["subject_id"] = df["subject_id"].apply(lambda x: ast.literal_eval(x))

    data = {
        "question_id": list(df["question_id"]),
        "subject_id": list(df["subject_id"]),
    }

    return data


def get_question_meta(beta):
    data = load_question_meta()
    unique = list(set([l[1] for l in data["subject_id"]]))
    subject_lst = [0 for _ in range(len(unique))]
    id_lst = data["question_id"]
    encoded_data = np.zeros(len(id_lst))
    for i in range(len(id_lst)):
        encoded_lst = subject_lst
        for j in range(len(subject_lst)):
            if data["subject_id"][i][1] == unique[j]:
                question_id = id_lst[i]
                encoded_data[question_id] = (j+1) * beta
    return encoded_data


def get_question_matrix():
    data = load_question_meta()

    q_matrix = np.zeros((388, 1774))
    for i, q_id in enumerate(data["question_id"]):
        for s_id in data["subject_id"]:
            q_matrix[s_id, q_id] = 1
    
    return q_matrix
import os

path = "F:\\Nasopharyn_Image\\train_data"
patients_no = []
for patient in os.listdir(path):
    patient_path = os.path.join(path, patient)
    dirs = os.listdir(patient_path)
    if len(dirs) < 4:
        patients_no.append(patient)
print(patients_no)
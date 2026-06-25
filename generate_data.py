import pandas as pd
import random

departments = [
    "Cardiology",
    "Neurology",
    "Orthopedics",
    "Pediatrics",
    "Oncology"
]

diagnosis = {
    "Cardiology": ["Hypertension", "Heart Disease", "Heart Failure"],
    "Neurology": ["Stroke", "Migraine", "Epilepsy"],
    "Orthopedics": ["Fracture", "Arthritis", "Back Pain"],
    "Pediatrics": ["Flu", "Infection", "Fever"],
    "Oncology": ["Lung Cancer", "Breast Cancer", "Leukemia"]
}

data = []

for i in range(1000):

    dept = random.choice(departments)

    data.append([
        f"P{i+1:04d}",
        random.randint(18, 85),
        random.choice(["Male", "Female"]),
        dept,
        random.choice(diagnosis[dept]),
        random.randint(1, 15),
        random.randint(500, 15000)
    ])

df = pd.DataFrame(
    data,
    columns=[
        "Patient_ID",
        "Age",
        "Gender",
        "Department",
        "Diagnosis",
        "Length_of_Stay",
        "Cost"
    ]
)

df.to_csv("patient_data.csv", index=False)

print("1000 patients generated")
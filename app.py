import tkinter as tk
from tkinter import messagebox
import pandas as pd
import joblib

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Load the trained machine learning pipeline
model = joblib.load("model.pkl")

# Feature names and order must match the training dataset
FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age"
]

# Create the main window
root = tk.Tk()
root.title("Diabetes Risk Assessment Dashboard")
root.geometry("850x750")
root.configure(bg="#F4F6F9")

# Title
title = tk.Label(
    root,
    text="Diabetes Risk Assessment Dashboard",
    font=("Arial", 20, "bold"),
    bg="#F4F6F9",
    fg="#243B53"
)
title.pack(pady=15)

subtitle = tk.Label(
    root,
    text="Educational machine-learning demonstration",
    font=("Arial", 10),
    bg="#F4F6F9",
    fg="#52606D"
)
subtitle.pack()

# Main form
form = tk.Frame(root, bg="white", padx=20, pady=15)
form.pack(pady=15)

# Input fields
fields = [
    ("Pregnancies", "Pregnancies"),
    ("Glucose level", "Glucose"),
    ("Blood pressure", "BloodPressure"),
    ("Skin thickness", "SkinThickness"),
    ("Insulin level", "Insulin"),
    ("BMI", "BMI"),
    ("Diabetes pedigree function", "DiabetesPedigreeFunction"),
    ("Age", "Age")
]

entries = {}

for row, (label_text, key) in enumerate(fields):
    label = tk.Label(
        form,
        text=label_text + ":",
        font=("Arial", 10),
        bg="white"
    )
    label.grid(row=row, column=0, padx=10, pady=6, sticky="w")

    entry = tk.Entry(form, width=22, font=("Arial", 10))
    entry.grid(row=row, column=1, padx=10, pady=6)
    entries[key] = entry

# Results area
result_label = tk.Label(
    root,
    text="Enter the measurements and select Assess Risk.",
    font=("Arial", 12, "bold"),
    bg="#F4F6F9",
    fg="#243B53",
    wraplength=750
)
result_label.pack(pady=10)

# Risk indicator
figure = Figure(figsize=(5, 1.6), dpi=100)
ax = figure.add_subplot(111)
ax.set_title("Model score (not a clinical diagnosis)")
ax.set_xlim(0, 100)
ax.set_ylim(0, 1)
ax.set_yticks([])
ax.set_xlabel("Model-estimated positive-class score (%)")

bar = ax.barh([0.5], [0], height=0.35)
ax.set_xticks([0, 25, 50, 75, 100])

canvas = FigureCanvasTkAgg(figure, master=root)
canvas.get_tk_widget().pack(pady=5)

def assess_risk():
    try:
        values = []

        for key in FEATURES:
            raw_value = entries[key].get().strip()

            if not raw_value:
                messagebox.showwarning(
                    "Missing input",
                    f"Please enter a value for {key}."
                )
                return

            value = float(raw_value)

            if value < 0:
                messagebox.showwarning(
                    "Invalid input",
                    f"{key} cannot be negative."
                )
                return

            values.append(value)

        # Check a few basic plausibility constraints
        if not values[0].is_integer():
            messagebox.showwarning(
                "Invalid input",
                "Pregnancies must be a whole number."
            )
            return

        if not values[7].is_integer() or values[7] <= 0:
            messagebox.showwarning(
                "Invalid input",
                "Age must be a positive whole number."
            )
            return

        if values[1] <= 0 or values[5] <= 0:
            messagebox.showwarning(
                "Invalid input",
                "Glucose and BMI must be greater than zero."
            )
            return

        # Build input in the exact order used during training
        input_df = pd.DataFrame([values], columns=FEATURES)

        prediction = int(model.predict(input_df)[0])
        probabilities = model.predict_proba(input_df)[0]
        positive_index = list(model.classes_).index(1)
        score = float(probabilities[positive_index]) * 100

        if prediction == 1:
            result_label.config(
                text=(
                    "Model output: Positive class\n"
                    "The model classified this input in the "
                    "dataset's positive class. This is not a diagnosis."
                ),
                fg="#B42318"
            )
        else:
            result_label.config(
                text=(
                    "Model output: Negative class\n"
                    "The model classified this input in the "
                    "dataset's negative class. This does not rule out diabetes."
                ),
                fg="#18794E"
            )

        # Update the visual indicator
        bar[0].set_width(score)
        ax.set_title(f"Model-estimated positive-class score: {score:.1f}%")
        canvas.draw()

    except ValueError:
        messagebox.showerror(
            "Invalid input",
            "Please enter valid numeric values in all fields."
        )
    except Exception as error:
        messagebox.showerror(
            "Error",
            f"Unable to assess the input:\n{error}"
        )

def reset_form():
    for entry in entries.values():
        entry.delete(0, tk.END)

    result_label.config(
        text="Enter the measurements and select Assess Risk.",
        fg="#243B53"
    )

    bar[0].set_width(0)
    ax.set_title("Model score (not a clinical diagnosis)")
    canvas.draw()

# Buttons
button_frame = tk.Frame(root, bg="#F4F6F9")
button_frame.pack(pady=10)

tk.Button(
    button_frame,
    text="Assess Risk",
    command=assess_risk,
    bg="#2563EB",
    fg="white",
    font=("Arial", 11, "bold"),
    width=15
).pack(side="left", padx=10)

tk.Button(
    button_frame,
    text="Reset",
    command=reset_form,
    bg="#D9E2EC",
    font=("Arial", 11),
    width=12
).pack(side="left", padx=10)

# Disclaimer
disclaimer = tk.Label(
    root,
    text=(
        "Educational prototype only. Not for diagnosis or treatment. "
        "Model scores may be poorly calibrated and may not generalise "
        "to other populations."
    ),
    font=("Arial", 9),
    bg="#F4F6F9",
    fg="#52606D",
    wraplength=780
)
disclaimer.pack(pady=10)

root.mainloop()

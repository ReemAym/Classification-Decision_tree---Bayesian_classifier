import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from tkinter import filedialog

from dt import DecisionTree
from naive import NaiveBayes

class GUIApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Diabetes Prediction")
        self.root.geometry("1000x700")

        # Create labels and entry widgets for input
        ttk.Label(root, text="Percentage of Data to Read (0-100):").grid(column=0, row=0, padx=10, pady=10)
        self.percentage_entry = ttk.Entry(root)
        self.percentage_entry.grid(column=1, row=0, padx=10, pady=10)

        ttk.Label(root, text="Percentage of Training Data (0-100):").grid(column=0, row=1, padx=10, pady=10)
        self.train_percentage_entry = ttk.Entry(root)
        self.train_percentage_entry.grid(column=1, row=1, padx=10, pady=10)

        ttk.Label(root, text="Percentage of Test Data (0-100):").grid(column=0, row=2, padx=10, pady=10)
        self.test_percentage_entry = ttk.Entry(root)
        self.test_percentage_entry.grid(column=1, row=2, padx=10, pady=10)

        # Create a browse button to choose the file
        self.browse_button = ttk.Button(root, text="Browse", command=self.browse_file)
        self.browse_button.grid(column=0, row=3, padx=10, pady=10)

        # Create a text field to display the chosen file path
        self.file_path_entry = ttk.Entry(root, width=50)
        self.file_path_entry.grid(column=1, row=3, padx=10, pady=10, columnspan=2)

        # Create a button to trigger the analysis
        self.analyze_button = ttk.Button(root, text="Analyze", command=self.analyze_data)
        self.analyze_button.grid(column=0, row=4, columnspan=2, pady=10)

        # Create a text widget to display the output
        self.output_text = tk.Text(root, height=28, width=120)
        self.output_text.grid(row=5, column=0, columnspan=3, padx=10, pady=10)

        # Add scrollbar to the output text widget
        self.scrollbar = tk.Scrollbar(root, command=self.output_text.yview)
        self.scrollbar.grid(row=5, column=3, sticky='nsew')
        self.output_text.config(yscrollcommand=self.scrollbar.set)

    def browse_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            self.file_path_entry.delete(0, tk.END)
            self.file_path_entry.insert(0, file_path)

    def analyze_data(self):
        try:
            percentage = float(self.percentage_entry.get())
            train_percentage = float(self.train_percentage_entry.get())
            test_percentage = float(self.test_percentage_entry.get())
            file_path = self.file_path_entry.get()

            # Read the CSV file
            data = pd.read_csv(file_path).sample(frac=percentage / 100)

            # Split dataset into features and target variable
            X = data.drop(columns=['diabetes']).values
            y = data['diabetes'].values

            # Split the dataset into training and test sets
            X_train_DT, X_test_DT, y_train_DT, y_test_DT = train_test_split(X, y, train_size=train_percentage / 100,
                                                                            test_size=test_percentage / 100)

            # Instantiate and fit the decision tree model
            tree = DecisionTree(max_depth=5)
            tree.fit(X_train_DT, y_train_DT)

            # Make predictions on the test data
            y_pred_DT = tree.predict(X_test_DT)

            # Compute the accuracy of the decision tree model
            accuracy_DT = accuracy_score(y_test_DT, y_pred_DT)

            # Instantiate and fit the Naive Bayes model
            naive_bayes = NaiveBayes()
            naive_bayes.fit(X_train_DT, y_train_DT)

            # Make predictions on the test data
            y_pred_N = naive_bayes.predict(X_test_DT)

            # Compute the accuracy of the Naive Bayes model
            accuracy_N = accuracy_score(y_test_DT, y_pred_N)

            if accuracy_N > accuracy_DT:
                better_model = "Naive Bayes"
            elif accuracy_N < accuracy_DT:
                better_model = "Decision Tree"
            else:
                better_model = "Both models have the same accuracy"

            # Prepare the predicted tuples string
            predicted_tuples = "\nPredicted Tuples:\n"
            for i in range(len(X_test_DT)):  # Display up to 5 predicted tuples
                predicted_tuples += f"Tuple {X_test_DT[i]}: Predicted class - Decision Tree: {y_pred_DT[i]}, Naive Bayes: {y_pred_N[i]}\n"

            # Prepare the output message
            output_message = f"Input Parameters:\nPercentage of Data to Read: {percentage}\nPercentage of Training Data: {train_percentage}\nPercentage of Test Data: {test_percentage}\n\nAccuracy of Decision Tree: {accuracy_DT}\nAccuracy of Naive Bayes: {accuracy_N}\nBetter Model: {better_model}\n{predicted_tuples}"

            # Clear the output text widget and insert the output message
            self.output_text.delete("1.0", tk.END)
            self.output_text.insert(tk.END, output_message)

        except ValueError:
            messagebox.showerror("Error", "Please enter valid percentages.")


if __name__ == "__main__":
    root = tk.Tk()
    app = GUIApp(root)
    root.mainloop()

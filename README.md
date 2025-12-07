# 🥑 Avocado Sales Prediction (ML + Flask)

This project predicts **weekly avocado sales volume (Total Volume)** using a **Machine Learning model** trained on the Avocado dataset.  
It also has a **Flask web app** frontend where the user can input:

- Date (week)
- Average price
- Avocado type (conventional / organic)
- Region

and get the **predicted weekly sales volume**.

---

## 🔧 Tech Stack

- **Python**
- **Pandas** – data loading & preprocessing  
- **Scikit-learn** – ML model (Random Forest Regressor)  
- **Flask** – web framework for frontend + backend  
- **HTML + Bootstrap** – simple UI

---

## 📂 Project Structure

```text
.
├── app.py                  # Flask web app
├── train_model.py          # Trains the ML model and saves it as .pkl
├── Avocado.csv             # Dataset
├── templates/
│   └── index.html          # Frontend page
└── README.md

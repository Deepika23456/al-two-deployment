# 🎓 Student Performance Predictor

Predicts a student's final exam score and grade from study time, attendance,
sleep, previous grade and background details.

🔗 **Live app:** https://al-two-deployment-ydp8actukappggy3f7sxgst.streamlit.app/

## Tech stack
Python, pandas, scikit-learn (Gradient Boosting), Streamlit

## Files
- `app.py`: Streamlit web app
- `model_utils.py`: model training and prediction
- `data_embedded.py`: embedded training dataset
- `requirements.txt`: dependencies

## Run locally
pip install -r requirements.txt
streamlit run app.py

## Model
Predicts `final_exam_score` (R2 about 0.58, MAE about 5 marks); the grade is derived from the score.

import streamlit as st

from model_utils import load_model, predict

st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓")


@st.cache_resource
def get_model():
    return load_model()


model = get_model()

st.title("🎓 Student Performance Predictor")
st.write("Enter the student's details to predict the final exam score and grade.")

col1, col2 = st.columns(2)
with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    study_time = st.slider("Study time (hours/day)", 0.0, 10.0, 3.5, 0.1)
    attendance = st.slider("Attendance (%)", 0.0, 100.0, 85.0, 0.5)
    sleep = st.slider("Sleep (hours/day)", 3.0, 10.0, 7.0, 0.1)
    previous_grade = st.slider("Previous grade", 0.0, 100.0, 70.0, 0.5)
with col2:
    parental_education = st.selectbox(
        "Parental education", ["High School", "Bachelors", "Masters", "PhD", "Unknown"]
    )
    internet = st.radio("Internet access", ["Yes", "No"], horizontal=True)
    extra = st.radio("Extracurricular activities", ["Yes", "No"], horizontal=True)
    job = st.radio("Part-time job", ["No", "Yes"], horizontal=True)

if st.button("Predict", type="primary"):
    result = predict(
        model,
        {
            "gender": gender,
            "study_time_hours": study_time,
            "attendance_percent": attendance,
            "sleep_hours": sleep,
            "parental_education": None if parental_education == "Unknown" else parental_education,
            "internet_access": internet,
            "extracurricular_activities": extra,
            "part_time_job": job,
            "previous_grade": previous_grade,
        },
    )
    m1, m2 = st.columns(2)
    m1.metric("Predicted final exam score", f"{result['predicted_score']} / 100")
    m2.metric("Predicted grade", result["predicted_grade"])
    st.caption("Model is trained on a 1,000-student dataset. Typical error is about ±5 marks.")

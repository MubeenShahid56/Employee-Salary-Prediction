import streamlit as st
import joblib
import pandas as pd
import os

# Resolve path relative to this script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "linearmodel.pkl")
model = joblib.load(model_path)

# Real-world verified average annual salaries (USD) — Statista 2024, worlddata.info, ABS, regional surveys
COUNTRY_AVG_SALARY = {
    'United States': 83000, 'Switzerland': 90000, 'Australia': 63000,
    'Canada': 55000, 'Germany': 57000, 'United Kingdom': 50000,
    'France': 46000, 'Japan': 38000, 'Singapore': 52000, 'Netherlands': 58000,
    'Saudi Arabia': 27000, 'United Arab Emirates': 30000, 'Qatar': 32000,
    'Kuwait': 25000, 'Bahrain': 20000, 'Oman': 18000, 'Turkey': 12000,
    'Brazil': 10000, 'Mexico': 10000, 'Malaysia': 12000, 'South Africa': 11000,
    'Jordan': 8000, 'Lebanon': 6000, 'Egypt': 5500, 'Morocco': 5000,
    'Indonesia': 5000, 'Philippines': 5500, 'India': 7000,
    'Pakistan': 4000, 'Nigeria': 4500, 'Kenya': 4000, 'Syria': 3000,
}

DEPARTMENTS = sorted([
    'Quality Control', 'Major Mfg Projects', 'Manufacturing', 'Product Development',
    'Sales', 'Account Management', 'Green Building', 'IT', 'Facilities/Engineering',
    'Marketing', 'Manufacturing Admin', 'Training', 'Quality Assurance',
    'Professional Training Group', 'Environmental Compliance', 'Creative',
    'Research/Development', 'Environmental Health/Safety', 'Human Resources',
    'Research Center', 'Finance', 'Legal', 'Operations', 'Customer Support',
    'Data Science', 'Cybersecurity', 'Supply Chain', 'Business Development',
    'AI Development', 'Machine Learning Engineering', 'Robotics',
    'Computer Vision', 'Natural Language Processing', 'AI Research',
    'Autonomous Systems', 'AI Ethics & Governance', 'Generative AI',
    'AI Infrastructure', 'Prompt Engineering', 'MLOps',
])

COUNTRIES = sorted(COUNTRY_AVG_SALARY.keys())

EDU_LEVELS = ['High School', 'Associate Degree', "Bachelor's Degree", "Master's Degree", 'PhD']
EMP_TYPES  = ['Full-Time', 'Part-Time', 'Contract', 'Freelance', 'Internship']

ALL_SKILLS = sorted([
    'Python', 'Java', 'SQL', 'Machine Learning', 'Data Analysis',
    'Project Management', 'Communication', 'Leadership', 'Excel', 'Power BI',
    'JavaScript', 'Cloud Computing', 'DevOps', 'Cybersecurity', 'Networking',
    'Deep Learning', 'TensorFlow', 'PyTorch', 'Computer Vision', 'NLP',
    'Reinforcement Learning', 'Robotics', 'ROS', 'Generative AI', 'LLMs',
    'Prompt Engineering', 'MLOps', 'Data Engineering', 'Statistics', 'R',
])

st.title("Employee Salary Prediction App")
st.caption("Predictions are based on real-world salary benchmarks (Statista 2024, worlddata.info, ABS, regional surveys).")
st.divider()

col1, col2 = st.columns(2)

with col1:
    country   = st.selectbox("Country", COUNTRIES, index=COUNTRIES.index('United States'))
    dept      = st.selectbox("Department", DEPARTMENTS, index=DEPARTMENTS.index('IT'))
    edu       = st.selectbox("Education Level", EDU_LEVELS, index=2)
    emp_type  = st.selectbox("Employment Type", EMP_TYPES)

with col2:
    years    = st.number_input("Years at Company", value=3, step=1, min_value=0, max_value=40)
    jobrate  = st.number_input("Job Rate (1.0 – 5.0)", value=3.0, step=0.5, min_value=1.0, max_value=5.0)
    overtime = st.number_input("Overtime Hours / Year", value=0, step=10, min_value=0, max_value=500)

    # Show real country average as reference
    avg = COUNTRY_AVG_SALARY.get(country, 0)
    st.metric("National Average Salary (all jobs)", f"${avg:,} / yr",
              help="Source: Statista 2024, worlddata.info, ABS, regional surveys")

st.subheader("Skills")
selected_skills = st.multiselect("Select applicable skills", options=ALL_SKILLS, default=['Communication'])
skill_count = len(selected_skills)
if skill_count == 0:
    st.warning("Select at least one skill for a more accurate prediction.")

st.divider()
predict = st.button("Predict Salary", use_container_width=True)
st.divider()

if predict:
    input_data = pd.DataFrame([{
        'Years':           years,
        'Job Rate':        jobrate,
        'Overtime Hours':  overtime,
        'Skill Count':     skill_count,
        'Department':      dept,
        'Country':         country,
        'Education Level': edu,
        'Employment Type': emp_type,
    }])

    try:
        prediction = model.predict(input_data)[0]
        st.balloons()

        col_a, col_b = st.columns(2)
        col_a.metric("Estimated Annual Salary", f"${prediction:,.0f}")
        col_b.metric("vs. National Average", f"${avg:,}",
                     delta=f"${prediction - avg:+,.0f}",
                     delta_color="normal")

        with st.expander("Prediction breakdown"):
            st.write(f"- Country: {country} (national avg: ${avg:,})")
            st.write(f"- Department: {dept}")
            st.write(f"- Education: {edu}")
            st.write(f"- Employment Type: {emp_type}")
            st.write(f"- Years at Company: {years}")
            st.write(f"- Job Rate: {jobrate}")
            st.write(f"- Overtime Hours: {overtime}")
            st.write(f"- Skills ({skill_count}): {', '.join(selected_skills) if selected_skills else 'None'}")
            st.caption("Note: Predictions are estimates based on statistical modelling of real salary benchmarks. "
                       "Actual salaries vary by employer, city, and individual negotiation.")
    except Exception as e:
        st.error(f"Prediction error: {e}")
else:
    st.info("Fill in the profile above and click 'Predict Salary'.")

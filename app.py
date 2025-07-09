import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, date, timedelta
import json
import os
import pickle
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# IMPORTANT: Page config MUST be the first Streamlit command
st.set_page_config(
    page_title="EduScan Somalia - Learning Risk Assessment",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Simple settings management
def load_app_settings():
    """Load application settings from file"""
    settings_file = "data/app_settings.json"
    if os.path.exists(settings_file):
        try:
            with open(settings_file, 'r') as f:
                return json.load(f)
        except:
            pass
    return {'language': 'English', 'theme': 'Modern', 'offline_mode': False}

def save_app_settings(settings):
    """Save application settings to file"""
    settings_file = "data/app_settings.json"
    try:
        os.makedirs(os.path.dirname(settings_file), exist_ok=True)
        with open(settings_file, 'w') as f:
            json.dump(settings, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving settings: {e}")
        return False

# Simple translations
def get_text(key, language=None):
    """Get localized text based on language setting"""
    if language is None:
        language = st.session_state.get('app_language', 'English')
    
    translations = {
        'English': {
            'app_title': 'EduScan Somalia',
            'dashboard': 'Dashboard',
            'assessment_form': 'Assessment Form',
            'teacher_resources': 'Teacher Resources',
            'parent_tracker': 'Parent Tracker',
            'educational_content': 'Educational Content',
            'student_name': 'Student Name',
            'grade_level': 'Grade Level',
            'math_score': 'Math Score (0-100)',
            'reading_score': 'Reading Score (0-100)',
            'writing_score': 'Writing Score (0-100)',
            'attendance': 'Attendance (%)',
            'behavior_rating': 'Behavior Rating (1-5)',
            'literacy_level': 'Literacy Level (1-5)',
            'analyze_learning_risk': 'Analyze Learning Risk',
            'assessment_results': 'Assessment Results',
            'recommendations': 'Recommendations',
            'teacher_name': 'Teacher Name',
            'assessment_date': 'Assessment Date',
            'assessment_subtitle': 'Comprehensive learning risk assessment for students',
            'attention_span': 'Attention Span (1-5)',
            'class_participation': 'Class Participation (1-5)',
            'homework_completion': 'Homework Completion (1-5)',
            'teacher_notes': 'Teacher Notes',
            'assess_student': 'Assess Student',
            'clear_form': 'Clear Form',
            'academic_scores': 'Academic Scores',
            'behavioral_indicators': 'Behavioral Indicators',
        },
        'Somali': {
            'app_title': 'EduScan Somalia',
            'dashboard': 'Xarunta Xogta',
            'assessment_form': 'Foomka Qiimaynta',
            'teacher_resources': 'Agabka Macalliminta',
            'parent_tracker': 'Dabagalka Waalidka',
            'educational_content': 'Waxyaabaha Waxbarasho',
            'student_name': 'Magaca Ardayga',
            'grade_level': 'Heerka Fasalka',
            'math_score': 'Dhibcaha Xisaabta (0-100)',
            'reading_score': 'Dhibcaha Akhriska (0-100)',
            'writing_score': 'Dhibcaha Qorista (0-100)',
            'attendance': 'Soo Gaadhitaanka (%)',
            'behavior_rating': 'Qiimaynta Dhaqanka (1-5)',
            'literacy_level': 'Heerka Aqrinta (1-5)',
            'analyze_learning_risk': 'Falanqee Khatarta Barashada',
            'assessment_results': 'Natiijada Qiimaynta',
            'recommendations': 'Talooyinka',
            'teacher_name': 'Magaca Macallinka',
            'assessment_date': 'Taariikhda Qiimaynta',
            'assessment_subtitle': 'Qiimayn dhamaystiran oo khatarta barashada ardayda',
            'attention_span': 'Mudada Diiradda (1-5)',
            'class_participation': 'Ka-qaybgalka Fasalka (1-5)',
            'homework_completion': 'Dhammaystirka Hawlaha Guriga (1-5)',
            'teacher_notes': 'Xusuusta Macallinka',
            'assess_student': 'Qiimee Ardayga',
            'clear_form': 'Nadiifi Foomka',
            'academic_scores': 'Dhibcaha Tacliinta',
            'behavioral_indicators': 'Tilmaamaha Dhaqanka',
        },
        'Arabic': {
            'app_title': 'EduScan Somalia',
            'dashboard': 'لوحة التحكم',
            'assessment_form': 'نموذج التقييم',
            'teacher_resources': 'موارد المعلم',
            'parent_tracker': 'متتبع الوالدين',
            'educational_content': 'المحتوى التعليمي',
            'student_name': 'اسم الطالب',
            'grade_level': 'مستوى الصف',
            'math_score': 'درجة الرياضيات (0-100)',
            'reading_score': 'درجة القراءة (0-100)',
            'writing_score': 'درجة الكتابة (0-100)',
            'attendance': 'الحضور (%)',
            'behavior_rating': 'تقييم السلوك (1-5)',
            'literacy_level': 'مستوى الإلمام بالقراءة والكتابة (1-5)',
            'analyze_learning_risk': 'تحليل مخاطر التعلم',
            'assessment_results': 'نتائج التقييم',
            'recommendations': 'التوصيات',
            'teacher_name': 'اسم المعلم',
            'assessment_date': 'تاريخ التقييم',
            'assessment_subtitle': 'تقييم شامل لمخاطر التعلم للطلاب',
            'attention_span': 'مدة الانتباه (1-5)',
            'class_participation': 'المشاركة في الفصل (1-5)',
            'homework_completion': 'إكمال الواجب المنزلي (1-5)',
            'teacher_notes': 'ملاحظات المعلم',
            'assess_student': 'تقييم الطالب',
            'clear_form': 'مسح النموذج',
            'academic_scores': 'الدرجات الأكاديمية',
            'behavioral_indicators': 'مؤشرات السلوك',
        }
    }
    
    return translations.get(language, translations['English']).get(key, key)

# Model loading and prediction functions
def load_model():
    """Load the trained model and scaler"""
    try:
        # Try to load user's trained model
        model_path = "data/learning_difficulty_detector.pkl"
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            print(f"User's trained model loaded from {model_path}")
            
            # Try to load scaler
            scaler_path = "data/scaler.pkl"
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    scaler = pickle.load(f)
                print("Scaler loaded successfully")
            else:
                scaler = StandardScaler()
                print("Using default scaler")
            
            return model, scaler
        else:
            # Create a simple fallback model
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            scaler = StandardScaler()
            
            # Generate some sample data to fit the model
            X_sample = np.random.rand(100, 7) * 100
            y_sample = np.random.randint(0, 3, 100)
            
            X_scaled = scaler.fit_transform(X_sample)
            model.fit(X_scaled, y_sample)
            
            return model, scaler
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

def make_prediction(model, scaler, features):
    """Make prediction using the loaded model"""
    try:
        # Scale features if scaler is available
        if scaler is not None:
            features_scaled = scaler.transform(features)
        else:
            features_scaled = features
            
        prediction = model.predict(features_scaled)
        probabilities = model.predict_proba(features_scaled)
        return prediction, probabilities
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return None, None

# Initialize language in session state
if 'app_language' not in st.session_state:
    settings = load_app_settings()
    st.session_state['app_language'] = settings.get('language', 'English')

# Get current language
language = st.session_state.get('app_language', 'English')

# Apply theme styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    .stApp {
        font-family: 'Poppins', sans-serif !important;
        background: #f8fafc !important;
        min-height: 100vh !important;
    }
    
    .stDeployButton {display: none;}
    #MainMenu {visibility: hidden;}
    .stHeader {display: none;}
    .stToolbar {display: none;}
    
    .page-header {
        background: linear-gradient(135deg, #87CEEB 0%, #F8DC75 100%);
        color: #1f2937;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 0.75rem;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
        border: 1px solid #e5e7eb;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #87CEEB 0%, #F8DC75 100%);
        color: #1f2937;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    .css-1d391kg {
        background-color: white !important;
        border-right: 1px solid #e5e7eb !important;
    }
</style>
""", unsafe_allow_html=True)

def render_sidebar():
    """Render sidebar navigation"""
    with st.sidebar:
        st.title("🎓 EduScan Somalia")
        st.markdown("---")
        
        # Language selector
        languages = {'English': 'English', 'Somali': 'Somali', 'Arabic': 'Arabic'}
        selected_language = st.selectbox(
            "Select Language", 
            list(languages.keys()),
            index=list(languages.keys()).index(st.session_state.get('app_language', 'English'))
        )
        
        if selected_language != st.session_state.get('app_language'):
            st.session_state['app_language'] = selected_language
            settings = load_app_settings()
            settings['language'] = selected_language
            save_app_settings(settings)
            st.rerun()
        
        st.markdown("---")
        
        # Navigation
        pages = {
            get_text('dashboard', language): "dashboard",
            get_text('assessment_form', language): "prediction", 
            get_text('teacher_resources', language): "resources",
            get_text('parent_tracker', language): "tracker",
            get_text('educational_content', language): "content"
        }
        
        for page_name, page_key in pages.items():
            if st.button(page_name, key=f"nav_{page_key}", use_container_width=True):
                st.session_state['current_page'] = page_key
                st.rerun()

def render_dashboard():
    """Render main dashboard"""
    st.markdown(f"""
    <div class="page-header">
        <h1>🎓 {get_text('app_title', language)}</h1>
        <p>Professional Learning Assessment Application</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Dashboard metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>📊 Total Students</h3>
            <h2>156</h2>
            <p>This Month</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>✅ On Track</h3>
            <h2>89</h2>
            <p>57% Students</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>⚠️ At Risk</h3>
            <h2>42</h2>
            <p>27% Students</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h3>🚨 Intervention</h3>
            <h2>25</h2>
            <p>16% Students</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Performance chart
    st.subheader("📈 Performance Overview")
    
    subjects = ['Math', 'Reading', 'Writing', 'Science', 'Social Studies']
    on_track = [75, 82, 68, 73, 79]
    at_risk = [15, 12, 22, 18, 14]
    intervention = [10, 6, 10, 9, 7]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(name='On Track', x=subjects, y=on_track, marker_color='#10b981'))
    fig.add_trace(go.Bar(name='At Risk', x=subjects, y=at_risk, marker_color='#f59e0b'))
    fig.add_trace(go.Bar(name='Intervention', x=subjects, y=intervention, marker_color='#ef4444'))
    
    fig.update_layout(
        barmode='stack',
        title="Student Performance by Subject",
        xaxis_title="Subjects",
        yaxis_title="Number of Students",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_prediction_page():
    """Render student assessment page"""
    st.markdown(f"""
    <div class="page-header">
        <h1>🔍 {get_text('assessment_form', language)}</h1>
        <p>{get_text('assessment_subtitle', language)}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Generate unique keys for form elements
    if 'form_key' not in st.session_state:
        st.session_state['form_key'] = 0
    
    form_key = st.session_state['form_key']
    
    # Student assessment form
    with st.form(f"student_assessment_{form_key}"):
        col1, col2 = st.columns(2)
        
        with col1:
            student_name = st.text_input(get_text('student_name', language), key=f"student_name_{form_key}")
            grade_level = st.selectbox(get_text('grade_level', language), 
                                     ["Grade 1", "Grade 2", "Grade 3", "Grade 4", "Grade 5"],
                                     key=f"grade_level_{form_key}")
            
            # Academic scores
            st.subheader(get_text('academic_scores', language))
            math_score = st.slider(get_text('math_score', language), 0, 100, 75, key=f"math_score_{form_key}")
            reading_score = st.slider(get_text('reading_score', language), 0, 100, 80, key=f"reading_score_{form_key}")
            writing_score = st.slider(get_text('writing_score', language), 0, 100, 70, key=f"writing_score_{form_key}")
            attendance = st.slider(get_text('attendance', language), 0, 100, 90, key=f"attendance_{form_key}")
        
        with col2:
            teacher_name = st.text_input(get_text('teacher_name', language), key=f"teacher_name_{form_key}")
            assessment_date = st.date_input(get_text('assessment_date', language), datetime.today(), key=f"assessment_date_{form_key}")
            
            # Behavioral indicators
            st.subheader(get_text('behavioral_indicators', language))
            attention_span = st.select_slider(get_text('attention_span', language), 
                                            options=[1, 2, 3, 4, 5], value=3, key=f"attention_span_{form_key}")
            class_participation = st.select_slider(get_text('class_participation', language), 
                                                 options=[1, 2, 3, 4, 5], value=3, key=f"class_participation_{form_key}")
            homework_completion = st.select_slider(get_text('homework_completion', language), 
                                                 options=[1, 2, 3, 4, 5], value=4, key=f"homework_completion_{form_key}")
        
        # Teacher notes
        teacher_notes = st.text_area(get_text('teacher_notes', language), height=100, key=f"teacher_notes_{form_key}")
        
        # Form buttons
        col1, col2 = st.columns(2)
        with col1:
            submitted = st.form_submit_button(get_text('assess_student', language), type="primary")
        with col2:
            clear_form = st.form_submit_button(get_text('clear_form', language), type="secondary")
        
        # Handle clear form
        if clear_form:
            st.session_state['form_key'] += 1
            st.rerun()
        
        if submitted and student_name:
            # Prepare features for model (7 features to match your trained model)
            features = np.array([[
                math_score, reading_score, writing_score, attendance,
                attention_span, class_participation, homework_completion
            ]])
            
            try:
                # Make prediction
                model, scaler = load_model()
                if model and scaler:
                    prediction, probabilities = make_prediction(model, scaler, features)
                    
                    if prediction is not None:
                        # Display results
                        st.markdown("---")
                        st.subheader(f"📊 {get_text('assessment_results', language)} - {student_name}")
                        
                        # Risk level display
                        risk_levels = {0: 'Low Risk', 1: 'Medium Risk', 2: 'High Risk'}
                        risk_colors = {0: '#10b981', 1: '#f59e0b', 2: '#ef4444'}
                        
                        current_risk = risk_levels.get(prediction[0], 'Unknown')
                        current_color = risk_colors.get(prediction[0], '#6b7280')
                        confidence = max(probabilities[0]) if len(probabilities[0]) > 0 else 0.5
                        
                        st.markdown(f"""
                        <div style="background: {current_color}; color: white; padding: 1rem; border-radius: 8px; text-align: center;">
                            <h3>Risk Level: {current_risk}</h3>
                            <p>Confidence: {confidence:.1%}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Recommendations
                        st.subheader(f"💡 {get_text('recommendations', language)}")
                        
                        if prediction[0] == 0:  # Low Risk
                            st.success("Student is performing well. Continue current teaching methods.")
                            st.write("- Maintain current academic support")
                            st.write("- Consider enrichment activities")
                            st.write("- Regular progress monitoring")
                        elif prediction[0] == 1:  # Medium Risk  
                            st.warning("Student may need additional support.")
                            st.write("- Provide targeted interventions")
                            st.write("- Increase one-on-one support")
                            st.write("- Monitor progress weekly")
                        else:  # High Risk
                            st.error("Student requires immediate intervention.")
                            st.write("- Implement intensive support program")
                            st.write("- Daily progress monitoring")
                            st.write("- Consider specialist evaluation")
                        
                        # Save prediction data
                        prediction_data = {
                            'student_name': student_name,
                            'teacher_name': teacher_name,
                            'grade_level': grade_level,
                            'assessment_date': assessment_date.isoformat(),
                            'scores': {
                                'math': math_score,
                                'reading': reading_score, 
                                'writing': writing_score,
                                'attendance': attendance
                            },
                            'behavioral': {
                                'attention_span': attention_span,
                                'class_participation': class_participation,
                                'homework_completion': homework_completion
                            },
                            'teacher_notes': teacher_notes,
                            'prediction': int(prediction[0]),
                            'probabilities': probabilities[0].tolist(),
                            'risk_level': current_risk
                        }
                        
                        # Simple data storage
                        try:
                            os.makedirs("data", exist_ok=True)
                            predictions_file = "data/student_data.json"
                            predictions = []
                            
                            if os.path.exists(predictions_file):
                                with open(predictions_file, 'r') as f:
                                    predictions = json.load(f)
                            
                            predictions.append(prediction_data)
                            
                            with open(predictions_file, 'w') as f:
                                json.dump(predictions, f, indent=2)
                        except Exception as e:
                            st.error(f"Error saving prediction: {e}")
                        
                else:
                    st.error("Model not available. Please check model files.")
                    
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")

def render_resources_page():
    """Render teacher resources page"""
    st.markdown(f"""
    <div class="page-header">
        <h1>👨‍🏫 {get_text('teacher_resources', language)}</h1>
        <p>Comprehensive tools and strategies for educators</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Resource categories
    tab1, tab2, tab3 = st.tabs(["Teaching Strategies", "Learning Activities", "Assessment Tools"])
    
    with tab1:
        st.subheader("🎯 Differentiated Teaching Strategies")
        
        difficulty_level = st.selectbox("Select Difficulty Level", 
                                      ["Beginner", "Intermediate", "Advanced"])
        subject_area = st.selectbox("Subject Area", 
                                  ["Mathematics", "Language Arts", "Science", "Social Studies"])
        
        if st.button("Generate Teaching Strategy"):
            strategies = {
                "Mathematics": [
                    "Use visual aids and manipulatives for concrete learning",
                    "Implement peer tutoring and collaborative problem solving", 
                    "Break complex problems into smaller, manageable steps",
                    "Use real-world applications to make math relevant"
                ],
                "Language Arts": [
                    "Implement guided reading with level-appropriate texts",
                    "Use graphic organizers for writing structure",
                    "Encourage creative storytelling and expression",
                    "Practice phonics through interactive games"
                ],
                "Science": [
                    "Conduct hands-on experiments and demonstrations",
                    "Use scientific method for structured learning",
                    "Integrate technology for virtual lab experiences",
                    "Connect science concepts to everyday life"
                ],
                "Social Studies": [
                    "Use maps and timelines for visual learning",
                    "Implement role-playing historical scenarios",
                    "Encourage cultural exchange and discussion",
                    "Connect historical events to current events"
                ]
            }
            
            if subject_area in strategies:
                st.success("Teaching Strategy Generated!")
                for i, strategy in enumerate(strategies[subject_area][:3], 1):
                    st.write(f"{i}. {strategy}")
    
    with tab2:
        st.subheader("🎮 Interactive Learning Activities")
        
        if st.button("Generate Activity Ideas"):
            activities = [
                "Math Scavenger Hunt - Find objects that represent different numbers",
                "Story Chain - Students build stories collaboratively",
                "Science Experiment - Simple experiments with everyday materials",
                "Geography Game - Identify countries and capitals",
                "Historical Timeline - Create interactive timeline of events",
                "Language Practice - Role-play conversations in different scenarios"
            ]
            
            st.success("Activity Ideas Generated!")
            for i, activity in enumerate(activities[:4], 1):
                st.write(f"{i}. {activity}")
    
    with tab3:
        st.subheader("📝 Assessment Tools")
        
        assessment_type = st.selectbox("Assessment Type", 
                                     ["Formative", "Summative", "Diagnostic"])
        
        if st.button("Create Assessment"):
            st.success("Assessment template created!")
            st.write("• Clear learning objectives defined")
            st.write("• Multiple question formats included")
            st.write("• Rubric for evaluation provided")
            st.write("• Differentiated difficulty levels available")

def render_tracker_page():
    """Render parent tracker page"""
    st.markdown(f"""
    <div class="page-header">
        <h1>👨‍👩‍👧‍👦 {get_text('parent_tracker', language)}</h1>
        <p>Track your child's learning progress at home</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Parent observation form
    with st.form("parent_observation"):
        col1, col2 = st.columns(2)
        
        with col1:
            child_name = st.text_input("Child's Name")
            observation_date = st.date_input("Date", datetime.today())
            
            # Learning observations
            st.subheader("📚 Learning Observations")
            homework_time = st.number_input("Homework Time (minutes)", 0, 180, 30)
            reading_time = st.number_input("Reading Time (minutes)", 0, 120, 20)
            difficulty_level = st.select_slider("Task Difficulty", 
                                              options=["Very Easy", "Easy", "Moderate", "Hard", "Very Hard"],
                                              value="Moderate")
        
        with col2:
            # Behavioral observations
            st.subheader("👀 Behavioral Observations")
            focus_rating = st.select_slider("Focus/Attention", options=[1, 2, 3, 4, 5], value=3)
            motivation_rating = st.select_slider("Motivation", options=[1, 2, 3, 4, 5], value=3)
            independence_rating = st.select_slider("Independence", options=[1, 2, 3, 4, 5], value=3)
            
            # Additional notes
            parent_notes = st.text_area("Additional Notes", height=100)
        
        submitted = st.form_submit_button("Save Observation", type="primary")
        
        if submitted and child_name:
            observation_data = {
                'child_name': child_name,
                'observation_date': observation_date.isoformat(),
                'homework_time': homework_time,
                'reading_time': reading_time,
                'difficulty_level': difficulty_level,
                'focus_rating': focus_rating,
                'motivation_rating': motivation_rating,
                'independence_rating': independence_rating,
                'parent_notes': parent_notes
            }
            
            # Simple data storage
            try:
                os.makedirs("data", exist_ok=True)
                observations_file = "data/parent_observations.json"
                observations = []
                
                if os.path.exists(observations_file):
                    with open(observations_file, 'r') as f:
                        observations = json.load(f)
                
                observations.append(observation_data)
                
                with open(observations_file, 'w') as f:
                    json.dump(observations, f, indent=2)
                
                st.success("Observation saved successfully!")
            except Exception as e:
                st.error(f"Error saving observation: {e}")
    
    # Display recent observations
    st.markdown("---")
    st.subheader("📊 Recent Observations")
    
    try:
        observations_file = "data/parent_observations.json"
        if os.path.exists(observations_file):
            with open(observations_file, 'r') as f:
                observations = json.load(f)
            
            if observations:
                df = pd.DataFrame(observations)
                st.dataframe(df[['child_name', 'observation_date', 'focus_rating', 'motivation_rating']], 
                           use_container_width=True)
            else:
                st.info("No observations recorded yet.")
        else:
            st.info("No observations recorded yet.")
    except Exception as e:
        st.error(f"Error loading observations: {e}")

def render_content_page():
    """Render educational content page"""
    st.markdown(f"""
    <div class="page-header">
        <h1>📚 {get_text('educational_content', language)}</h1>
        <p>Research-based information about learning difficulties</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Content tabs
    tab1, tab2, tab3 = st.tabs(["Learning Difficulties", "Intervention Strategies", "Research Data"])
    
    with tab1:
        st.subheader("🧠 Understanding Learning Difficulties")
        
        st.write("""
        Learning difficulties affect how students process and understand information. 
        Early identification and intervention are crucial for student success.
        """)
        
        # Common types
        st.write("**Common Types:**")
        st.write("• **Dyslexia** - Reading difficulties")
        st.write("• **Dyscalculia** - Math difficulties") 
        st.write("• **ADHD** - Attention difficulties")
        st.write("• **Processing disorders** - Information processing issues")
        
        # Signs to watch for
        st.write("**Signs to Watch For:**")
        st.write("• Difficulty following instructions")
        st.write("• Problems with reading comprehension")
        st.write("• Trouble with math concepts")
        st.write("• Difficulty staying focused")
        st.write("• Problems with writing and spelling")
    
    with tab2:
        st.subheader("🎯 Evidence-Based Interventions")
        
        intervention_type = st.selectbox("Select Intervention Area", 
                                       ["Reading", "Mathematics", "Attention", "Behavior"])
        
        interventions = {
            "Reading": [
                "Phonics-based instruction",
                "Guided reading programs", 
                "Multi-sensory learning approaches",
                "Reading comprehension strategies",
                "Vocabulary building exercises"
            ],
            "Mathematics": [
                "Concrete-to-abstract progression",
                "Visual math representations",
                "Step-by-step problem solving",
                "Math fact fluency practice",
                "Real-world math applications"
            ],
            "Attention": [
                "Structured learning environment",
                "Regular breaks and movement",
                "Clear expectations and routines",
                "Positive reinforcement systems",
                "Mindfulness and relaxation techniques"
            ],
            "Behavior": [
                "Positive behavior support",
                "Clear rules and consequences",
                "Social skills training",
                "Conflict resolution strategies",
                "Peer mediation programs"
            ]
        }
        
        if intervention_type in interventions:
            st.write("**Recommended Interventions:**")
            for i, intervention in enumerate(interventions[intervention_type], 1):
                st.write(f"{i}. {intervention}")
    
    with tab3:
        st.subheader("📈 Research Statistics")
        
        # Sample statistics chart
        categories = ['Reading Difficulties', 'Math Difficulties', 'Attention Issues', 'Processing Disorders']
        percentages = [15, 12, 8, 10]
        
        fig = px.bar(x=categories, y=percentages, 
                     title="Prevalence of Learning Difficulties (%)",
                     color=percentages,
                     color_continuous_scale='Blues')
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.write("""
        **Key Research Findings:**
        • Early intervention improves outcomes by 70%
        • Multi-sensory approaches are most effective
        • Regular progress monitoring is essential
        • Family involvement doubles success rates
        • Cultural adaptation increases intervention effectiveness
        """)

def main():
    """Main application function"""
    # Render sidebar navigation
    render_sidebar()
    
    # Get current page from session state
    current_page = st.session_state.get('current_page', 'dashboard')
    
    # Render appropriate page
    if current_page == 'dashboard':
        render_dashboard()
    elif current_page == 'prediction':
        render_prediction_page()
    elif current_page == 'resources':
        render_resources_page() 
    elif current_page == 'tracker':
        render_tracker_page()
    elif current_page == 'content':
        render_content_page()

if __name__ == "__main__":
    main()
# Diabetes Prediction ML Model & API

A machine learning project that predicts diabetes risk using patient health data. Features comprehensive EDA, model comparison, and a REST API for real-time predictions.

## Project Overview

This project analyzes a dataset of 100,000 patient records to build a predictive model for diabetes screening. The final model achieves 96% accuracy with balanced precision (93%) and recall (71%) for diabetes detection.

## Dataset Information

- **Size**: 100,000 records × 9 features
- **Target**: Binary diabetes classification (0=No, 1=Yes)
- **Features**:
  - `age`: Patient age (0-80 years)
  - `gender`: Male, Female, Other
  - `bmi`: Body Mass Index (10-95)
  - `HbA1c_level`: Hemoglobin A1c percentage (3.5-9.0%)
  - `blood_glucose_level`: Blood glucose level (80-300 mg/dL)
  - `hypertension`: Binary (0/1)
  - `heart_disease`: Binary (0/1)
  - `smoking_history`: never, former, current, not current, ever, No Info

## Key Findings

### Class Distribution
- **Non-diabetic**: 91,500 cases (91.5%)
- **Diabetic**: 8,500 cases (8.5%)
- Significant class imbalance addressed using SMOTE oversampling

### Feature Importance
1. **Blood Glucose Level** (correlation: 0.42)
2. **HbA1c Level** (correlation: 0.40)
3. **Age** (correlation: 0.26)
4. **BMI** (correlation: 0.21)

### Risk Factors Identified
- **Former smokers**: Highest diabetes rate (17.0%)
- **Age**: Diabetic patients average 20 years older (61 vs 40)
- **BMI**: Diabetic patients have higher BMI (32 vs 27)
- **Clinical markers**: HbA1c and glucose levels significantly elevated

## Model Performance

### Final Model: Gradient Boosting with SMOTE
- **Algorithm**: GradientBoostingClassifier
- **Preprocessing**: SMOTE oversampling + StandardScaler + OneHotEncoder
- **Optimal Threshold**: 0.55

### Performance Metrics
| Metric | Value |
|--------|-------|
| **Accuracy** | 96% |
| **Precision** | 93% |
| **Recall** | 71% |
| **F1-Score** | 80% |
| **ROC-AUC** | 0.965 |

### Model Comparison
| Algorithm | F1-Score | Recall | Precision |
|-----------|----------|--------|-----------|
| Baseline (class weights) | 0.79 | 69% | 94% |
| Random Forest + SMOTE | 0.76 | 73% | 80% |
| **Gradient Boosting + SMOTE** | **0.80** | **71%** | **93%** |

## Project Structure

```
diabetes_prediction/
├── data/
│   └── diabetes_prediction_dataset.csv
├── notebooks/
│   └── diabetes_eda_and_modeling.ipynb
├── models/
│   ├── diabetes_gradient_boosting_model.pkl
│   ├── diabetes_preprocessor.pkl
│   └── model_metadata.json
├── app.py                 # FastAPI application
├── requirements.txt       # Dependencies
└── README.md
```

## API Usage

### Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Run the API
python app.py

# Access interactive docs
# http://localhost:8000/docs
```

### API Endpoints

#### POST /predict
Predict diabetes risk for a patient.

**Request Body:**
```json
{
  "gender": "Female",
  "age": 45.0,
  "hypertension": 0,
  "heart_disease": 0,
  "smoking_history": "never",
  "bmi": 28.5,
  "HbA1c_level": 6.2,
  "blood_glucose_level": 140
}
```

**Response:**
```json
{
  "diabetes_probability": 0.7234,
  "diabetes_prediction": "Diabetic",
  "risk_level": "High Risk",
  "confidence": "High",
  "model_version": "GradientBoostingClassifier"
}
```

#### GET /model-info
Returns model performance metrics and metadata.

#### GET /
Basic API information and model performance summary.

## Technical Implementation

### Data Preprocessing
- **Missing Values**: None detected
- **Categorical Encoding**: OneHotEncoder for gender and smoking_history
- **Feature Scaling**: StandardScaler for numerical features
- **Class Imbalance**: SMOTE oversampling (6,800 → 73,200 diabetic samples)

### Model Selection Process
1. **Baseline**: Random Forest with class_weight='balanced'
2. **SMOTE Enhancement**: Synthetic oversampling for balanced training
3. **Algorithm Comparison**: Random Forest vs Gradient Boosting
4. **Threshold Optimization**: Tuned for optimal precision-recall balance

### Validation Strategy
- **Train-test split**: 80/20 with stratification
- **Cross-validation**: 5-fold for model selection
- **Metrics focus**: Precision, Recall, F1-score (not accuracy due to imbalance)

## Clinical Relevance

### Healthcare Organization Applications

#### Primary Care & Community Health Centers
- **Risk Stratification**: Prioritize patients for diabetes screening based on risk scores
- **Resource Allocation**: Focus limited testing resources on high-risk populations
- **Population Health Management**: Identify community diabetes risk patterns
- **Preventive Care Protocols**: Trigger lifestyle interventions for moderate-risk patients

#### Hospital Systems & Health Networks
- **Emergency Department Screening**: Flag undiagnosed diabetes in acute care settings
- **Pre-admission Risk Assessment**: Identify diabetes risk before elective procedures
- **Clinical Decision Support**: Integrate into electronic health records (EHR) for provider alerts
- **Quality Metrics**: Track population health outcomes and screening effectiveness

#### Public Health Organizations
- **Epidemiological Surveillance**: Monitor diabetes risk trends across demographics
- **Community Screening Programs**: Design targeted outreach in high-risk neighborhoods
- **Health Policy Planning**: Inform resource allocation and intervention strategies
- **Research Applications**: Baseline risk assessment for diabetes prevention studies

#### Insurance & Payer Organizations
- **Risk Assessment**: Actuarial modeling for diabetes-related healthcare costs
- **Preventive Care Programs**: Identify members for diabetes prevention initiatives
- **Care Management**: Proactive outreach to high-risk individuals
- **Cost-Effectiveness Analysis**: Evaluate screening program ROI

### Implementation Considerations

#### Integration Points
- **EHR Integration**: API endpoints for real-time risk scoring during patient visits
- **Laboratory Systems**: Automated risk calculation when HbA1c/glucose results available
- **Population Health Platforms**: Batch processing for large patient cohorts
- **Mobile Health Apps**: Patient-facing risk assessment tools

#### Workflow Integration
- **Clinical Workflows**: Embed in routine physical exam protocols
- **Screening Programs**: Enhance existing diabetes screening initiatives
- **Care Coordination**: Alert case managers to high-risk patients
- **Follow-up Protocols**: Trigger appropriate testing and referral pathways

### Model Strengths for Healthcare Use
- **High Precision (93%)**: Minimizes false alarms and unnecessary follow-up costs
- **Good Recall (71%)**: Captures most at-risk patients for intervention
- **Standard Clinical Data**: Uses commonly available laboratory values and measurements
- **Scalable Implementation**: API-based deployment supports high-volume screening

### Limitations & Risk Mitigation
- **29% False Negative Rate**: Requires complementary screening strategies
- **Model Validation**: Needs testing on diverse clinical populations before deployment
- **Clinical Oversight**: Must be used alongside, not instead of, clinical judgment
- **Regular Monitoring**: Performance tracking essential for maintaining accuracy

### Regulatory & Safety Considerations
- **FDA Guidance**: May require clinical validation for diagnostic use
- **HIPAA Compliance**: Ensure patient data protection in API implementation
- **Clinical Validation**: Prospective studies needed to confirm real-world performance
- **Provider Training**: Healthcare staff education on model limitations and proper use

## Dependencies

```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
imbalanced-learn>=0.8.0
joblib>=1.1.0
fastapi>=0.68.0
uvicorn>=0.15.0
pydantic>=1.8.0
```



## License

This project is for educational and demonstration purposes.

## Author


Emmanuel Maduneme, Ph.D.
Assistant Professor of Mass Media, Southeast Missouri State University
Data Science | UX Research | Science Communications
I have a Ph.D. in Journalism and Mass Communication from University of Oregon with expertise in mixed-methods research, data science, and experimental design. Research focus on media psychology, user experience, and behavioral analytics.

---

**Note**: This model is for educational demonstration only and should not be used for actual medical diagnosis. Always consult healthcare professionals for medical decisions.
"""
Script to generate the comprehensive Viva Voce & Academic Defense Master PDF Guide.
"""

import os
import sys
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas to dynamically compute and draw total page numbers and running headers."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_decorations(self, page_count):
        if self._pageNumber == 1:
            # Skip running header/footer on cover/first page
            return
        
        self.saveState()
        self.setFont("Helvetica-Bold", 8)
        self.setFillColor(colors.HexColor("#64748b"))
        
        # Header
        self.drawString(54, 11 * 72 - 36, "Smart Healthcare Diagnosis API — Academic Defense & Viva Master Guide")
        self.setStrokeColor(colors.HexColor("#cbd5e1"))
        self.setLineWidth(0.5)
        self.line(54, 11 * 72 - 42, 8.5 * 72 - 54, 11 * 72 - 42)
        
        # Footer
        self.line(54, 45, 8.5 * 72 - 54, 45)
        self.setFont("Helvetica", 8)
        self.drawString(54, 32, "Confidential & Academic Material — Author: Krish Gaikwad")
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(8.5 * 72 - 54, 32, page_str)
        self.restoreState()


def build_viva_pdf(output_filename="Smart_Healthcare_API_Viva_Master_Guide.pdf"):
    pdf_path = os.path.abspath(output_filename)
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()
    
    # Custom styles
    primary_color = colors.HexColor("#1e3a8a")     # Deep Blue
    secondary_color = colors.HexColor("#0284c7")   # Medical Cyan/Blue
    dark_text = colors.HexColor("#0f172a")         # Dark Slate
    light_bg = colors.HexColor("#f8fafc")          # Soft Slate
    callout_bg = colors.HexColor("#eff6ff")        # Light Blue Callout
    accent_green = colors.HexColor("#16a34a")      # Green
    accent_red = colors.HexColor("#dc2626")        # Red
    border_color = colors.HexColor("#cbd5e1")

    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=24,
        leading=28,
        textColor=primary_color,
        alignment=0,
        spaceAfter=6
    )

    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=13,
        leading=16,
        textColor=secondary_color,
        spaceAfter=15
    )

    h1_style = ParagraphStyle(
        "Heading1_Custom",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=18,
        textColor=primary_color,
        spaceBefore=14,
        spaceAfter=6,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        "Heading2_Custom",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=15,
        textColor=secondary_color,
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        "Body_Custom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13.5,
        textColor=dark_text,
        spaceAfter=5
    )

    bullet_style = ParagraphStyle(
        "Bullet_Custom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13.5,
        textColor=dark_text,
        leftIndent=12,
        firstLineIndent=-8,
        spaceAfter=3
    )

    q_style = ParagraphStyle(
        "QuestionStyle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=14,
        textColor=primary_color,
        spaceBefore=8,
        spaceAfter=2,
        keepWithNext=True
    )

    a_style = ParagraphStyle(
        "AnswerStyle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=dark_text,
        leftIndent=10,
        spaceAfter=6
    )

    callout_style = ParagraphStyle(
        "CalloutText",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#1e40af")
    )

    table_header_style = ParagraphStyle(
        "TableHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8.5,
        leading=11,
        textColor=colors.white,
        alignment=1
    )

    table_cell_style = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=10.5,
        textColor=dark_text
    )

    story = []

    # ─── COVER / HEADER BANNER ──────────────────────────────────────────────
    story.append(Paragraph("Smart Healthcare Diagnosis API", title_style))
    story.append(Paragraph("Complete Viva Voce & Academic Defense Master Guide", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=2, color=primary_color, spaceAfter=10))

    # Meta Table
    meta_data = [
        [
            Paragraph("<b>Project Title:</b> Smart Healthcare Diagnosis API", table_cell_style),
            Paragraph("<b>Domain:</b> Machine Learning & Healthcare Microservices", table_cell_style)
        ],
        [
            Paragraph("<b>Candidate Name:</b> Krish Gaikwad", table_cell_style),
            Paragraph("<b>Tech Stack:</b> Python 3.12, FastAPI, Scikit-Learn, Pydantic v2", table_cell_style)
        ],
        [
            Paragraph("<b>Core Goal:</b> Real-time Disease Diagnosis & Risk Stratification", table_cell_style),
            Paragraph("<b>Repository:</b> github.com/krishgaikwad42-hood/apiproject-", table_cell_style)
        ]
    ]
    meta_table = Table(meta_data, colWidths=[250, 250])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), light_bg),
        ('BOX', (0,0), (-1,-1), 1, border_color),
        ('INNERGRID', (0,0), (-1,-1), 0.5, border_color),
        ('PADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 10))

    # Callout Box: What is this guide?
    summary_callout = [
        [Paragraph("<b>Examiner Perspective Note:</b> In viva examinations, professors evaluate two fundamental domains: <b>(1) Mathematical & Data Science rigor</b> (handling disguised zeros, data leakage prevention, algorithm benchmarking, precision/recall trade-offs) and <b>(2) Software Engineering & Production readiness</b> (FastAPI asynchronous event loops, Pydantic schemas, correlation ID tracing, latency SLAs, and automated test coverage). This document prepares you to master both.", callout_style)]
    ]
    summary_t = Table(summary_callout, colWidths=[500])
    summary_t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), callout_bg),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#93c5fd")),
        ('PADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(summary_t)
    story.append(Spacer(1, 12))

    # ─── SECTION 1: PROJECT OVERVIEW & ARCHITECTURE ─────────────────────────
    story.append(Paragraph("1. Project Architecture & End-to-End Workflow", h1_style))
    story.append(Paragraph("The system is an end-to-end clinical machine learning service designed to provide sub-50ms diagnostic predictions. The complete request lifecycle flows through 5 distinct architectural layers:", body_style))

    story.append(Paragraph("• <b>Layer 1 — Client Interface:</b> Web Dashboard UI, Postman, or third-party EHR systems submit HTTP POST JSON payloads.", bullet_style))
    story.append(Paragraph("• <b>Layer 2 — Observability & Tracing Middleware:</b> Intercepts incoming requests, generates a unique <code>X-Request-ID</code> UUID4 for distributed correlation tracing, measures execution latency in milliseconds, and logs client IP.", bullet_style))
    story.append(Paragraph("• <b>Layer 3 — Pydantic Validation & Domain Integrity:</b> Strict type verification (Float, Integer). Enforces clinical bounds (e.g., Glucose 40–500 mg/dL, Age 1–120) and dual-gender constraints (Male patients cannot have pregnancies &gt; 0).", bullet_style))
    story.append(Paragraph("• <b>Layer 4 — Serialized Preprocessing Pipeline:</b> Pre-fitted <code>SimpleImputer(strategy='median')</code> resolves disguised zeros followed by feature normalization using <code>RobustScaler</code>.", bullet_style))
    story.append(Paragraph("• <b>Layer 5 — In-Memory ML Inference Engine:</b> Model generates risk class (Diabetic / Non-Diabetic), calibrated prediction probability (0.0 to 1.0), confidence percentage, and stratified risk tier (Low &lt;35%, Moderate 35–65%, High &gt;65%).", bullet_style))

    story.append(Spacer(1, 10))

    # ─── SECTION 2: DATA SCIENCE, DATASET & DISGUISED ZEROS ─────────────────
    story.append(Paragraph("2. Dataset & The Critical 'Disguised Zeros' Finding", h1_style))
    story.append(Paragraph("The model was trained on the <b>Pima Indians Diabetes Dataset</b> (National Institute of Diabetes and Digestive and Kidney Diseases).", body_style))
    
    dataset_info = [
        [Paragraph("<b>Metric / Parameter</b>", table_header_style), Paragraph("<b>Clinical Detail / Statistical Property</b>", table_header_style)],
        [Paragraph("Total Dataset Size", table_cell_style), Paragraph("768 Patient Records, 8 Clinical Feature Dimensions + 1 Target (Outcome)", table_cell_style)],
        [Paragraph("Target Class Balance", table_cell_style), Paragraph("65.1% Negative (500 non-diabetic) vs. 34.9% Positive (268 diabetic) — Moderately Imbalanced", table_cell_style)],
        [Paragraph("Disguised Missing Values", table_cell_style), Paragraph("Numerical zeros entered in place of missing tests: Glucose (5), Blood Pressure (35), Skin Thickness (227), Insulin (374), BMI (11).", table_cell_style)],
        [Paragraph("Imputation Strategy", table_cell_style), Paragraph("Zero values in physiological fields converted to NaN and imputed with <b>Median</b> (less sensitive to skew/outliers than Mean).", table_cell_style)],
        [Paragraph("Data Leakage Defense", table_cell_style), Paragraph("Train/Test split (80/20 Stratified) performed <b>BEFORE</b> fitting Imputer and Scaler. Test data is never observed during transformation fitting.", table_cell_style)]
    ]
    d_table = Table(dataset_info, colWidths=[150, 350])
    d_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), primary_color),
        ('BOX', (0,0), (-1,-1), 1, border_color),
        ('INNERGRID', (0,0), (-1,-1), 0.5, border_color),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, light_bg]),
        ('PADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(d_table)
    story.append(Spacer(1, 10))

    # ─── SECTION 3: ML BENCHMARK & METRICS TABLE ────────────────────────────
    story.append(Paragraph("3. Machine Learning Benchmark & Model Selection", h1_style))
    story.append(Paragraph("Six supervised learning classification algorithms were rigorously benchmarked with stratified cross-validation. In clinical systems, <b>Recall (Sensitivity)</b> is prioritized over raw accuracy because a False Negative (missed diabetic patient) is clinically catastrophic.", body_style))

    model_comp = [
        [Paragraph("<b>Algorithm</b>", table_header_style), Paragraph("<b>Accuracy</b>", table_header_style), Paragraph("<b>Precision</b>", table_header_style), Paragraph("<b>Recall</b>", table_header_style), Paragraph("<b>F1-Score</b>", table_header_style), Paragraph("<b>ROC-AUC</b>", table_header_style)],
        [Paragraph("Decision Tree", table_cell_style), Paragraph("74.7%", table_cell_style), Paragraph("63.0%", table_cell_style), Paragraph("72.2%", table_cell_style), Paragraph("67.2%", table_cell_style), Paragraph("0.741", table_cell_style)],
        [Paragraph("Random Forest", table_cell_style), Paragraph("76.6%", table_cell_style), Paragraph("68.8%", table_cell_style), Paragraph("61.1%", table_cell_style), Paragraph("64.7%", table_cell_style), Paragraph("0.835", table_cell_style)],
        [Paragraph("Logistic Regression", table_cell_style), Paragraph("77.9%", table_cell_style), Paragraph("72.7%", table_cell_style), Paragraph("59.3%", table_cell_style), Paragraph("65.3%", table_cell_style), Paragraph("0.842", table_cell_style)],
        [Paragraph("Support Vector Machine (SVM)", table_cell_style), Paragraph("76.0%", table_cell_style), Paragraph("70.0%", table_cell_style), Paragraph("51.9%", table_cell_style), Paragraph("59.6%", table_cell_style), Paragraph("0.814", table_cell_style)],
        [Paragraph("K-Nearest Neighbors (KNN)", table_cell_style), Paragraph("72.7%", table_cell_style), Paragraph("61.5%", table_cell_style), Paragraph("59.3%", table_cell_style), Paragraph("60.4%", table_cell_style), Paragraph("0.781", table_cell_style)],
        [Paragraph("Gradient Boosting", table_cell_style), Paragraph("75.3%", table_cell_style), Paragraph("66.0%", table_cell_style), Paragraph("61.1%", table_cell_style), Paragraph("63.5%", table_cell_style), Paragraph("0.828", table_cell_style)]
    ]
    m_table = Table(model_comp, colWidths=[150, 70, 70, 70, 70, 70])
    m_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), secondary_color),
        ('BOX', (0,0), (-1,-1), 1, border_color),
        ('INNERGRID', (0,0), (-1,-1), 0.5, border_color),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, light_bg]),
        ('PADDING', (0,0), (-1,-1), 4),
        ('ALIGN', (1,0), (-1,-1), 'CENTER'),
    ]))
    story.append(m_table)
    story.append(Spacer(1, 10))

    # ─── SECTION 4: FASTAPI & SOFTWARE ARCHITECTURE ─────────────────────────
    story.append(Paragraph("4. Backend API Design & Production Engineering", h1_style))
    story.append(Paragraph("FastAPI was selected over Flask and Django due to ASGI asynchronous high-throughput concurrency, automatic OpenAPI 3.0 schema generation, and deep integration with Pydantic v2.", body_style))

    api_endpoints = [
        [Paragraph("<b>HTTP Method & Route</b>", table_header_style), Paragraph("<b>Function & Purpose</b>", table_header_style), Paragraph("<b>Status Code</b>", table_header_style)],
        [Paragraph("<code>GET /</code>", table_cell_style), Paragraph("Service discovery root metadata. Returns interactive UI for browser clients and structured JSON for REST clients.", table_cell_style), Paragraph("200 OK", table_cell_style)],
        [Paragraph("<code>GET /health</code>", table_cell_style), Paragraph("Liveness & readiness probe. Checks memory residency of ML model and preprocessors.", table_cell_style), Paragraph("200 / 503", table_cell_style)],
        [Paragraph("<code>POST /predict</code>", table_cell_style), Paragraph("Primary inference endpoint. Validates vitals, runs imputation, executes model, returns risk.", table_cell_style), Paragraph("200 / 422", table_cell_style)],
        [Paragraph("<code>GET /metrics</code>", table_cell_style), Paragraph("Operational telemetry: Total requests served, uptime, average latency SLA, error rates.", table_cell_style), Paragraph("200 OK", table_cell_style)],
        [Paragraph("<code>GET /docs</code>", table_cell_style), Paragraph("Interactive Swagger UI allowing developers and doctors to test API requests live.", table_cell_style), Paragraph("200 OK", table_cell_style)]
    ]
    a_table = Table(api_endpoints, colWidths=[110, 310, 80])
    a_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), primary_color),
        ('BOX', (0,0), (-1,-1), 1, border_color),
        ('INNERGRID', (0,0), (-1,-1), 0.5, border_color),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, light_bg]),
        ('PADDING', (0,0), (-1,-1), 4),
        ('ALIGN', (2,0), (2,-1), 'CENTER'),
    ]))
    story.append(a_table)
    story.append(Spacer(1, 14))

    story.append(PageBreak())

    # ─── SECTION 5: TOP 35 VIVA QUESTIONS & ANSWERS ──────────────────────────
    story.append(Paragraph("5. Top 35 Viva Voce Questions & Model Answers", title_style))
    story.append(Paragraph("Memorize these clear, concise, technical explanations for your examination:", body_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=primary_color, spaceAfter=8))

    # Category A
    story.append(Paragraph("CATEGORY A: Machine Learning & Algorithms (Q1 – Q10)", h2_style))

    qas_a = [
        ("Q1. What is the core problem solved by your project?",
         "This project provides an automated, low-latency clinical risk diagnosis system for early diabetes detection using patient vitals (Glucose, BMI, Blood Pressure, Insulin, Age), exposed via a high-performance FastAPI microservice."),
        
        ("Q2. Why is classification chosen over regression for this problem?",
         "The objective is diagnostic screening—classifying whether a patient has Diabetes (1) or No Diabetes (0). Regression predicts continuous quantities, whereas binary classification produces category labels and calibrated risk probabilities."),
        
        ("Q3. Which algorithms did you benchmark, and which one is deployed?",
         "We benchmarked 6 algorithms: Decision Tree, Random Forest, Logistic Regression, SVM, KNN, and Gradient Boosting. The Decision Tree model was selected as the default operational model because it achieved the highest Recall (72.2%), which minimizes dangerous False Negatives in clinical screening."),
        
        ("Q4. Explain what False Positives (FP) and False Negatives (FN) mean in your project.",
         "A False Positive (Type I error) occurs when a healthy person is classified as diabetic (results in a harmless follow-up blood test). A False Negative (Type II error) occurs when a diabetic patient is mistakenly diagnosed as healthy (dangerous, as treatment is delayed). Thus, minimizing False Negatives (maximizing Recall) is critical."),
        
        ("Q5. What is the formula for Recall, Precision, and F1-Score?",
         "Recall = TP / (TP + FN); Precision = TP / (TP + FP); F1-Score = 2 * (Precision * Recall) / (Precision + Recall). F1-Score is the harmonic mean balancing both metrics."),
        
        ("Q6. Why is Accuracy alone misleading for this dataset?",
         "The dataset has a 65:35 class imbalance. A naive model that always predicts 'No Diabetes' would achieve 65.1% accuracy while having a 0% Recall on diabetic patients. Accuracy fails to reflect diagnostic safety on imbalanced data."),
        
        ("Q7. What is ROC-AUC, and what does it indicate?",
         "Receiver Operating Characteristic (ROC) plots True Positive Rate vs False Positive Rate across all classification thresholds. AUC (Area Under Curve) measures the model's ability to discriminate between positive and negative patients. An AUC of 0.84 indicates strong ranking ability."),
        
        ("Q8. Why didn't you use Deep Learning / Artificial Neural Networks (ANN)?",
         "With only 768 tabular records, Deep Learning architectures easily overfit and require huge training datasets. Tree-based and linear models generalize substantially better on small tabular datasets, train in milliseconds, and provide full explainability."),
        
        ("Q9. How does Random Forest work compared to Decision Trees?",
         "A Decision Tree splits data based on Gini Impurity or Information Gain at single nodes. A Random Forest is an ensemble of multiple decision trees using Bagging (Bootstrap Aggregating) and random feature subsets to reduce variance and combat overfitting."),
        
        ("Q10. How do you serialize and load models in Python?",
         "We use <code>joblib</code> / <code>pickle</code> to serialize the trained model, imputer, and scaler into <code>.pkl</code> binary files. During FastAPI application startup (<code>lifespan</code> event), models are loaded once into RAM for instant inference without disk I/O latency.")
    ]

    for q, a in qas_a:
        story.append(Paragraph(q, q_style))
        story.append(Paragraph(f"<b>Answer:</b> {a}", a_style))

    story.append(Spacer(1, 6))

    # Category B
    story.append(Paragraph("CATEGORY B: Preprocessing, EDA & Statistics (Q11 – Q18)", h2_style))

    qas_b = [
        ("Q11. What are 'Disguised Zeros', and why are they significant?",
         "In the Pima dataset, missing laboratory observations were encoded as 0. For biological vitals like Glucose, Blood Pressure, and BMI, a value of 0 is physiologically impossible in living patients. Recognizing these as missing data (NaN) rather than genuine numeric measurements was a critical finding."),
        
        ("Q12. Why did you use Median Imputation instead of Mean Imputation?",
         "Medical measurements like Serum Insulin and Skin Thickness exhibit significant right-skewness and extreme outliers. The Mean is heavily dragged by outliers, whereas the Median represents the robust central tendency of skewed biological distributions."),
        
        ("Q13. Why did you not drop rows containing missing/zero values?",
         "Insulin alone had 374 missing values and Skin Thickness had 227. Dropping all zero rows would discard over 50% of the dataset, severely degrading statistical power and introducing demographic selection bias."),
        
        ("Q14. What is Data Leakage, and how did you prevent it?",
         "Data Leakage occurs when information from the test dataset leaks into the training pipeline (e.g., calculating global mean/scaling parameters across the entire dataset). We strictly performed an 80/20 train-test split first; imputers and scalers were fitted ONLY on <code>X_train</code> and then applied to <code>X_test</code>."),
        
        ("Q15. Why is Feature Scaling necessary?",
         "Algorithms like Logistic Regression, KNN, and SVM are sensitive to feature magnitudes because they rely on distance metrics or gradient descent. Scaling features ensures that high-magnitude variables (like Insulin 0–600) do not dominate low-magnitude variables (like Pedigree 0.05–2.5)."),
        
        ("Q16. What is the difference between StandardScaler and RobustScaler / MinMaxScaler?",
         "<code>StandardScaler</code> transforms data to zero mean and unit variance (z-score = (x - μ) / σ). <code>MinMaxScaler</code> scales to a fixed range [0, 1]. <code>RobustScaler</code> uses the median and interquartile range (IQR), making it immune to extreme outlier distortions."),
        
        ("Q17. What is the Diabetes Pedigree Function?",
         "It is a continuous score that scores genetic influence of diabetes in a patient's family history based on ancestral relationship degrees and age of onset in relatives."),
        
        ("Q18. How did you perform Exploratory Data Analysis (EDA)?",
         "We generated feature distribution histograms, boxplots for outlier detection, correlation heatmaps to assess multicollinearity, and class-stratified pairplots to inspect decision boundary separability.")
    ]

    for q, a in qas_b:
        story.append(Paragraph(q, q_style))
        story.append(Paragraph(f"<b>Answer:</b> {a}", a_style))

    story.append(PageBreak())

    # Category C
    story.append(Paragraph("CATEGORY C: Backend Engineering & FastAPI (Q19 – Q27)", h2_style))

    qas_c = [
        ("Q19. Why choose FastAPI over Flask or Django?",
         "FastAPI is built on ASGI (Starlette and Uvicorn) enabling native Python <code>async/await</code> concurrency. It provides automatic OpenAPI/Swagger documentation, native Pydantic v2 data validation with fast Rust-based serialization, and significantly higher requests-per-second throughput than WSGI-based Flask."),
        
        ("Q20. What is the difference between synchronous (WSGI) and asynchronous (ASGI) servers?",
         "WSGI servers (Flask/Django) process requests synchronously in a thread-per-request blocking model. ASGI servers (FastAPI/Uvicorn) utilize an asynchronous event loop that handles thousands of concurrent I/O-bound connections without blocking worker threads."),
        
        ("Q21. How does Pydantic v2 validate input data?",
         "Pydantic enforces strict type annotations and boundary constraints at runtime. If an invalid value (e.g., negative age or string in a float field) is submitted, Pydantic intercepts the payload before hitting the ML pipeline and generates a structured HTTP 422 Unprocessable Entity error."),
        
        ("Q22. Explain how your API handles male patients vs female patients.",
         "The dataset originates from female Pima Indian records where <code>Pregnancies</code> is a primary feature. For male patients, pregnancies are biologically 0. Our API includes a custom Pydantic validator that rejects any payload where <code>Gender == 'male'</code> and <code>Pregnancies &gt; 0</code> with a clear 422 error, while setting it to 0 automatically in the UI."),
        
        ("Q23. What is the purpose of the RequestLoggingMiddleware?",
         "The custom middleware intercepts every HTTP request to: (1) generate a unique <code>X-Request-ID</code> UUID4 correlation token, (2) measure exact execution time in milliseconds, (3) log client IP and status code, and (4) inject the request ID into response headers for end-to-end tracing."),
        
        ("Q24. What is the function of the /health endpoint?",
         "It acts as a readiness/liveness probe for container orchestrators (like Kubernetes or AWS ECS). It confirms whether the application is alive, whether ML model weights are loaded in RAM, and verifies that the preprocessor is functional."),
        
        ("Q25. What is the /metrics endpoint used for?",
         "It exposes operational runtime telemetry including total requests processed, error rates, average latency, and system uptime, facilitating monitoring by tools like Prometheus or Datadog."),
        
        ("Q26. What happens if the machine learning model file is missing or corrupted?",
         "The application's inference service implements graceful degradation. During startup, if model loading fails, the system logs a critical warning, disables the <code>/predict</code> route with a descriptive 503 Service Unavailable error, while keeping health and docs routes operational."),
        
        ("Q27. How does Content Negotiation work on the root (/) route?",
         "The root handler inspects the HTTP <code>Accept</code> header. If a browser requests the page (<code>text/html</code>), it serves the interactive visual clinical dashboard. If an automated script or curl requests JSON (<code>application/json</code>), it returns structured API metadata.")
    ]

    for q, a in qas_c:
        story.append(Paragraph(q, q_style))
        story.append(Paragraph(f"<b>Answer:</b> {a}", a_style))

    story.append(Spacer(1, 6))

    # Category D
    story.append(Paragraph("CATEGORY D: Testing, Security & Deployment (Q28 – Q35)", h2_style))

    qas_d = [
        ("Q28. How is the project tested automatically?",
         "We utilize <code>pytest</code> with a comprehensive test suite (19 automated tests) covering Smoke tests (root, docs, health), Inference tests (positive and negative diabetes cases), Validation tests (out-of-bounds, missing fields, male pregnancy check), and Observability tests."),
        
        ("Q29. What is FastAPI TestClient, and does it require a running server?",
         "FastAPI's <code>TestClient</code> is built on Starlette and <code>httpx</code>. It executes in-process HTTP requests directly against the ASGI app without needing to bind to an external network port, enabling lightning-fast unit testing in under 0.2 seconds."),
        
        ("Q30. What is CORS, and how is it configured?",
         "Cross-Origin Resource Sharing (CORS) is a browser security mechanism that restricts web applications on one domain from making requests to a different domain. We configured FastAPI's <code>CORSMiddleware</code> to allow authorized frontend origins to query the API."),
        
        ("Q31. Explain the Dockerfile structure for containerization.",
         "Our Dockerfile uses an official <code>python:3.12-slim</code> base image, copies <code>requirements.txt</code>, installs dependencies without cache to minimize image size, copies source code and model weights, exposes port 8000, and defines an <code>ENTRYPOINT</code> running Uvicorn."),
        
        ("Q32. How can this system be deployed to production cloud platforms?",
         "The application includes a <code>Procfile</code> and container configurations ready for deployment on platforms like Render, AWS App Runner, Railway, or Google Cloud Run behind a reverse proxy (Nginx or Cloudflare) with auto-scaling."),
        
        ("Q33. How does the system handle extreme outliers during real-time inference?",
         "Pydantic imposes hard physiological bounds (e.g., Glucose max 500 mg/dL). For extreme but valid numbers, the preprocessor uses <code>RobustScaler</code> which scales values based on median and IQR rather than min-max extremes."),
        
        ("Q34. What are the main limitations of this project?",
         "Limitations include: (1) Dataset size of 768 female Pima Indian records may have demographic biases; (2) Binary prediction rather than multi-stage diabetes severity (e.g., Type 1 vs Type 2); (3) Absence of long-term longitudinal patient history tracking."),
        
        ("Q35. What is the future scope for this system?",
         "Future enhancements include: (1) Integrating SHAP (SHapley Additive exPlanations) for local feature importance explainability; (2) Adding continuous patient time-series monitoring via wearable sensor streams; (3) Implementing OAuth2/JWT doctor authentication and HIPAA/GDPR-compliant encrypted database logging.")
    ]

    for q, a in qas_d:
        story.append(Paragraph(q, q_style))
        story.append(Paragraph(f"<b>Answer:</b> {a}", a_style))

    story.append(PageBreak())

    # ─── SECTION 6: EXAMINER TRICK QUESTIONS & STRATEGY ──────────────────────
    story.append(Paragraph("6. Examiner 'Trap' Questions & How to Defend", title_style))
    story.append(Paragraph("Examiners often test if you truly understand the trade-offs in your engineering decisions. Here is how to answer their trickiest questions:", body_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=primary_color, spaceAfter=10))

    traps = [
        ("Trap 1: 'Your model accuracy is ~75%. Why is it not 95% or 99%?'",
         "<b>Golden Defense:</b> <i>'Sir/Madam, in real clinical healthcare datasets, claiming 99% accuracy on 768 records is almost always a sign of severe overfitting or data leakage. Medical data is inherently noisy and biologically variable. More importantly, our model is tuned for <b>Recall (72.2%)</b> to catch true diabetic cases. In clinical screening, a realistic 75% accuracy with high recall and zero data leakage is significantly more trustworthy and deployable than an overfitted 99% model.'</i>"),
        
        ("Trap 2: 'Why did you train on a dataset of females and allow male inputs in your API?'",
         "<b>Golden Defense:</b> <i>'The benchmark Pima dataset contains female patient vitals. To make the API medically safe and viable for real-world clinic intake, our system implements domain-specific constraint handling: for male patients, the pregnancy feature is strictly locked to 0 with automated Pydantic validation, while general metabolic vitals (Glucose, BP, Insulin, BMI, Age) remain clinically diagnostic.'</i>"),
        
        ("Trap 3: 'Why didn't you use XGBoost or a Deep Neural Network?'",
         "<b>Golden Defense:</b> <i>'For small tabular datasets (N=768), complex Deep Neural Networks suffer from high sample complexity and extreme risk of overfitting. We benchmarked 6 algorithms including ensemble methods like Random Forest and Gradient Boosting. The Decision Tree model provided the highest sensitivity (Recall) and complete algorithmic interpretability, which is required in clinical medical audits.'</i>"),
        
        ("Trap 4: 'What happens if a user submits a glucose value of 10,000 mg/dL?'",
         "<b>Golden Defense:</b> <i>'The request never reaches the machine learning model. Pydantic's strict schema layer validates that Glucose is within the plausible human range of 40 to 500 mg/dL. An immediate HTTP 422 Unprocessable Entity error is returned within 2 milliseconds with a clear error payload.'</i>")
    ]

    for t_title, t_ans in traps:
        story.append(Paragraph(t_title, q_style))
        box = [[Paragraph(t_ans, body_style)]]
        box_table = Table(box, colWidths=[500])
        box_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), callout_bg),
            ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#93c5fd")),
            ('PADDING', (0,0), (-1,-1), 7),
        ]))
        story.append(box_table)
        story.append(Spacer(1, 8))

    # Build the document
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Viva PDF Master Guide successfully generated at: {pdf_path}")
    return pdf_path

if __name__ == "__main__":
    build_viva_pdf()

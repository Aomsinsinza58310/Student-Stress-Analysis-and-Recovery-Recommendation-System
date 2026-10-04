# ผลการทดลองเชิงประจักษ์ (Empirical Feasibility Study)
## สำหรับบรรจุในบทที่ 5: วิธีดำเนินการวิจัยและผลการทดลองนำร่อง

อ้างอิงชุดข้อมูลและแนวทางจากงานวิจัย: *G. Mahalakshmi & G. Sujatha, "Predictions of College Students’ Mental Stress using Machine Learning Algorithms", IEEE ICICCS 2023 (DOI: 10.1109/ICICCS56967.2023.10142672)*

---

### 5.3 การคัดเลือกชุดข้อมูลและการวิเคราะห์ข้อมูลสำรวจ (Ass 4: Dataset & EDA)

#### 5.3.1 รายละเอียดชุดข้อมูลสาธารณะที่เลือกใช้
โครงงานนี้ใช้ชุดข้อมูลมาตรฐาน **Student Stress Factors: A Comprehensive Analysis** (Kaggle Dataset) ซึ่งเป็นชุดข้อมูลที่สอดคล้องกับระเบียบวิธีวิจัยของ *Mahalakshmi & Sujatha (2023)* โดยมีคุณลักษณะเชิงปริมาณและสถิติพื้นฐาน ดังนี้:

- **จำนวนแถวข้อมูล (Instances):** 1,100 ตัวอย่าง
- **จำนวนตัวแปรต้น (Input Features):** 20 ตัวแปร
- **ตัวแปรเป้าหมาย (Target Label):** `stress_level` แบ่งออกเป็น 3 ระดับ:
  - `0`: ความเครียดระดับต่ำ (Low Stress)
  - `1`: ความเครียดระดับปานกลาง (Moderate Stress)
  - `2`: ความเครียดระดับสูง (High Stress)
- **กลุ่มปัจจัย 5 ด้าน (5 Multidimensional Factors):**
  1. *Psychological:* `anxiety_level`, `self_esteem`, `mental_health_history`, `depression`
  2. *Physiological:* `headache`, `blood_pressure`, `sleep_quality`, `breathing_problem`
  3. *Environmental:* `noise_level`, `living_conditions`, `safety`, `basic_needs`
  4. *Academic:* `academic_performance`, `study_load`, `teacher_student_relationship`, `future_career_concerns`
  5. *Social:* `social_support`, `peer_pressure`, `extracurricular_activities`, `bullying`

#### 5.3.2 สถิติเชิงบรรยายและการกระจายตัวของข้อมูล (EDA Plots)

1. **การแจกแจงของคลาสเป้าหมาย (Class Distribution):**
   - ระดับ 0 (Low Stress): 373 ตัวอย่าง (33.9%)
   - ระดับ 1 (Moderate Stress): 358 ตัวอย่าง (32.5%)
   - ระดับ 2 (High Stress): 369 ตัวอย่าง (33.5%)
   > สัดส่วนข้อมูลมีความสมดุลอย่างสมบูรณ์ (Balanced Dataset) จึงไม่จำเป็นต้องใช้เทคนิค Oversampling (SMOTE) เพิ่มเติม และไม่มีปัญหา Class Imbalance Bias ในการเทรนเบื้องต้น

![Target Class Distribution](eda_class_distribution.png)

2. **การกระจายตัวของตัวแปรสำคัญ (Feature Distribution across Stress Levels):**
   - ตัวแปร `anxiety_level` และ `depression` มีแนวโน้มสูงขึ้นอย่างมีนัยสำคัญตามระดับความเครียด
   - ในขณะที่ `sleep_quality` (คุณภาพการนอน) และ `academic_performance` มีค่าลดต่ำลงอย่างเด่นชัดในกลุ่มที่มีความเครียดสูง

![Feature Distributions](eda_feature_distributions.png)

3. **การวิเคราะห์สหสัมพันธ์ (Correlation Analysis):**
   - แผนภาพ Correlation Heatmap แสดงให้เห็นว่า ปัจจัยที่มีสหสัมพันธ์ทางบวกกับ `stress_level` สูงสุด ได้แก่ `bullying` (\(r = 0.75\)), `future_career_concerns` (\(r = 0.74\)), `anxiety_level` (\(r = 0.74\)), `depression` (\(r = 0.73\)) และ `headache` (\(r = 0.71\))
   - ปัจจัยที่มีสหสัมพันธ์ทางลบ (ปัจจัยปกป้อง/ลดทอนความเครียด) ได้แก่ `self_esteem` (\(r = -0.76\)), `sleep_quality` (\(r = -0.75\)), `safety` (\(r = -0.72\)) และ `basic_needs` (\(r = -0.71\))

![Correlation Heatmap](eda_correlation_heatmap.png)

#### 5.3.3 การเตรียมข้อมูลและการจัดการค่าสูญหาย (Data Preparation & Quality)
- การตรวจสอบไม่พบค่าสูญหาย (Missing Values = 0 แถว) และไม่พบค่าผิดปกติ (Outliers) ชนิดรุนแรง
- ดำเนินการแบ่งชุดข้อมูลเป็น **Train Set 80% (880 ตัวอย่าง)** และ **Test Set 20% (220 ตัวอย่าง)** โดยใช้เทคนิค *Stratified Sampling* เพื่อรักษาการกระจายตัวของคลาสเป้าหมายให้เหมือนกันทั้งสองส่วน

---

### 5.4 ผลการทดลองและประเมินประสิทธิภาพโมเดลเบื้องต้น (Ass 5: Model Evaluation)

#### 5.4.1 ผลการเปรียบเทียบประสิทธิภาพโมเดล (Performance Benchmark)
ทำการทดสอบประสิทธิภาพของอัลกอริทึมจำแนกประเภท 3 รูปแบบหลัก ได้แก่ **Random Forest (RF)**, **Support Vector Machine (SVM with RBF Kernel)** และ **eXtreme Gradient Boosting (XGBoost)** บน Test Set ที่ไม่เคยผ่านการเรียนรู้มาก่อน:

##### ตารางที่ 5.1: ผลการประเมินประสิทธิภาพโมเดลจำแนกความเครียด 3 ระดับ (Multi-class Classification)
| Model | Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) | Precision (Weighted) | Recall (Weighted) | F1-Score (Weighted) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | **89.09%** | **89.26%** | **89.13%** | **89.09%** | **89.26%** | **89.09%** | **89.07%** |
| **SVM (RBF Kernel)** | 88.64% | 89.35% | 88.65% | 88.67% | 89.39% | 88.64% | 88.68% |
| **XGBoost** | 87.73% | 87.79% | 87.75% | 87.74% | 87.79% | 87.73% | 87.72% |

##### ตารางที่ 5.2: ผลการประเมินเพื่อการคัดกรองกลุ่มเสี่ยงความเครียดสูง (High Stress vs Normal/Mild, Threshold = 0.50)
| Model | Accuracy | Precision (PPV) | Recall (Sensitivity) | Specificity | NPV | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | **92.73%** | **93.94%** | **83.78%** | **97.26%** | **92.21%** | **88.57%** |
| **SVM (RBF)** | 92.27% | 90.14% | 86.49% | 95.21% | 93.29% | 88.28% |
| **XGBoost** | 91.36% | 87.67% | 86.49% | 93.84% | 93.20% | 87.07% |

#### 5.4.2 ผลการปรับค่าเกณฑ์ความน่าจะเป็น (Probability Threshold Tuning)
ตามบริบทของระบบคัดกรองสุขภาพจิต การเกิดความผิดพลาดแบบผลลบลวง (False Negative: นักศึกษาเครียดสูงแต่โมเดลทำนายว่าไม่เสี่ยง) ก่อให้เกิดอันตรายสูงสุด จึงได้ทำการปรับลดค่าเกณฑ์ตัดสิน (Decision Threshold) จากเดิม 0.50 ลงมาที่ระดับ 0.10–0.20:

##### ตารางที่ 5.3: ผลการปรับ Threshold สำหรับ Random Forest ในการตรวจจับความเครียดระดับสูง
| Threshold | Sensitivity (Recall) | Specificity | Precision (PPV) | NPV | F1-Score | False Negatives (FN) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0.50 | 83.78% | 97.26% | 93.94% | 92.21% | 88.57% | 12 เคส |
| 0.40 | 87.84% | 93.15% | 86.67% | 93.79% | 87.25% | 9 เคส |
| 0.30 | 98.65% | 89.73% | 82.95% | 99.24% | 90.12% | 1 เคส |
| **0.20** | **100.00%** | **84.25%** | **76.29%** | **100.00%** | **86.55%** | **0 เคส** |
| 0.10 | 100.00% | 83.56% | 75.51% | 100.00% | 86.05% | 0 เคส |

> **ข้อค้นพบสำคัญ (Key Finding):** เมื่อปรับ Decision Threshold ลงมาที่ **0.20** ค่า **Sensitivity พุ่งขึ้นแตะ 100%** และค่า **Negative Predictive Value (NPV) เท่ากับ 100%** โดยสามารถกำจัดเคสหลุดรอด (False Negatives) จากเดิม 12 คน เหลือ **0 คน** โดยที่ Specificity ยังคงรักษาระดับสูงถึง 84.25% ซึ่งเป็นไปตามสมมติฐานทางคลินิกอย่างสมบูรณ์

![Threshold Tuning Tradeoff](threshold_tuning_tradeoff.png)

#### 5.4.3 ผลการวิเคราะห์อิทธิพลของตัวแปรด้วย SHAP (Explainable AI: XAI)
เพื่อตอบโจทย์ความโปร่งใสและอธิบายผลลัพธ์ของโมเดลได้ ได้ประยุกต์ใช้ **SHAP (SHapley Additive exPlanations)** บน Tree-based Model เพื่อตรวจสอบค่าน้ำหนักอิทธิพลของแต่ละปัจจัย:

![SHAP Summary Plot](shap_summary_plot.png)

- **Top Risk Factors:** ปัจจัยที่ส่งผลบวกต่อการทำนายความเครียดสูงที่สุดคือ `future_career_concerns` (ความกังวลด้านอาชีพในอนาคต), `bullying` (การถูกกลั่นแกล้ง), `anxiety_level` (ระดับความวิตกกังวล) และ `depression` (ภาวะซึมเศร้า)
- **Top Protective Factors:** ปัจจัยที่มี SHAP Value เป็นลบอย่างมีนัยสำคัญเมื่อมีค่าสูง ได้แก่ `self_esteem` (ความภาคภูมิใจในตนเอง), `sleep_quality` (คุณภาพการนอน) และ `safety` (ความปลอดภัยในชีวิตประจำวัน)

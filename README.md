# ECG Arrhythmia Classification with Knowledge Distillation

This project implements a Knowledge Distillation (KD) framework for ECG arrhythmia classification using the MIT-BIH Arrhythmia Database. A complex teacher CNN model is trained to classify ECG segments as normal or abnormal, and its knowledge is distilled into a lightweight student CNN model suitable for resource-constrained devices like wearables.

## Dataset
- **Source**: MIT-BIH Arrhythmia Database (PhysioNet)
- **Records Used**: 100, 101, 102
- **Task**: Binary classification (Normal vs. Abnormal heartbeats)

## Methodology
1. **Data Preprocessing**: ECG segments are extracted (200 samples per heartbeat), normalized, and split into train/validation/test sets.
2. **Teacher Model**: A complex CNN with ~500K parameters.
3. **Student Model**: A lightweight CNN with ~10K parameters.
4. **Knowledge Distillation**: The student learns from the teacher’s soft predictions and true labels using a combined loss function.
5. **Evaluation**: Models are compared using precision, recall, and F1-score on the test set.

## Results
- Classification reports are saved in `classification_report.txt`.
- Teacher model parameters: ~500K
- Student model parameters: ~10K
- Parameter reduction: ~90%

## Files
- `teacher_model.h5`: Saved teacher model
- `student_model.h5`: Saved student model
- `classification_report.txt`: Evaluation results
- `notebooks/`: Colab notebook (upload your .ipynb file)
- `src/`: Python scripts (e.g., `setup.py`, `load_data.py`, etc.)

## How to Run
1. Clone the repository.
2. Install dependencies: `pip install wfdb numpy pandas scikit-learn tensorflow matplotlib`
3. Run the notebook or scripts in the `src/` folder.

## Future Work
- Extend to multi-class classification.
- Deploy the student model on edge devices.
- Explore other datasets like PTB-XL.

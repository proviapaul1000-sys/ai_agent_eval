# ML Pipeline RL Training Task: Advanced Emotion Classification with Automatic Imbalance Detection

### Problem Statement

You are a machine learning engineer at a fast-growing startup providing emotion detection services for social media platforms. You need to build a robust text classification pipeline with automatic class imbalance detection that can be easily adapted to different datasets. Current ML pipelines load datasets without analyzing class distribution, apply uniform training approaches regardless of data characteristics, and evaluate using accuracy metrics that mask poor performance on rare or underrepresented classes. This results in models that achieve good overall accuracy but completely miss critical classes - causing production failures. The production environment uses mixed hardware (CPU-only and GPU-enabled servers), but no solution adapts model selection to available hardware or automatically handles class imbalance severity. You will build this robust pipeline using the `mteb/emotion` dataset as a test case to demonstrate its capabilities.

### Business Impact

Building a robust, reusable pipeline with automatic class imbalance detection directly addresses production failures that are costing user trust and revenue. The `mteb/emotion` dataset serves as an ideal test case as it contains significant class imbalance between majority and rare classes - exactly the scenario causing production issues. A pipeline that automatically detects imbalance (≥2:1 ratios) and applies severity-appropriate techniques will measurably improve performance on underrepresented classes while maintaining computational efficiency. This robust solution can then be easily adapted to other text classification datasets facing similar imbalance challenges. The impact is quantifiable through F1-macro scores on rare classes and the pipeline's adaptability to new datasets.

### Proposal

Build a robust, reusable text classification pipeline with automatic class imbalance detection. Use the `mteb/emotion` dataset as your test case to demonstrate the pipeline's capabilities - this dataset contains class imbalance that the system should automatically detect and handle appropriately. The pipeline must be designed for easy adaptation to other text classification datasets while handling missing data splits by creating stratified 8:1:1 splits. Support multiple modeling approaches: classical ML and transformer-based approaches, using larger models when GPU is available and lightweight models for CPU environments. The solution must compare approaches (classical ML, LLM prompting, transformer fine-tuning), select evaluation metrics based on detected data characteristics, and create an inference pipeline that uses the best-performing model for text classification on new samples.

## Requirements

- You must create a function to automatically analyze class distribution and classify imbalance severity using thresholds: Mild (2-3:1), Moderate (3-5:1), Severe (5-10:1), or Extreme (>10:1), without requiring manual configuration.

- When imbalance is detected (ratio ≥2:1), the pipeline must use F1-macro as the primary evaluation metric; for balanced datasets (ratio <2:1), accuracy must be used as the primary metric.

- You must implement functionality to handle missing data splits by creating train/validation/test splits in 8:1:1 ratio while maintaining class distribution across all splits.

- You must create a function to detect available hardware acceleration (GPU) and return appropriate modeling recommendations based on the detected hardware capabilities.

- Imbalance handling techniques must be applied conditionally based on severity: Mild uses class weighting, Moderate combines sampling with class weighting, Severe adds ensemble methods, and Extreme implements comprehensive multi-technique approaches.

- The pipeline must support both classical ML models and transformer approaches, automatically selecting between DistilBERT for CPU environments and BERT-base when GPU is available.

- You must implement comprehensive evaluation that includes per-class precision, recall, F1-scores, confusion matrix, and the appropriate primary metric (F1-macro or accuracy) based on imbalance detection results.

- You must create functionality to evaluate and compare at least three modeling approaches: classical ML (e.g., SVM, Random Forest), LLM prompting (zero-shot/few-shot), and transformer fine-tuning, demonstrating performance improvements on rare classes compared to baseline approaches.

- The pipeline must be designed for easy adaptation to other text classification datasets with minimal code modifications required.

- The inference pipeline must select the best-performing model based on evaluation results and provide a simple interface for making emotion predictions on new text samples, returning both predicted emotion labels and confidence scores.


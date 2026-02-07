"""
Data Quality Checker - Détection de problèmes de qualité des données.
Inclut: data leakage, missing values, outliers, distribution shift.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
from scipy import stats

from config.logging_config import get_logger

logger = get_logger(__name__)


class DataQualityChecker:
    """
    Checker complet pour la qualité des données ML.
    Détecte: leakage, missing values, outliers, distribution shifts.
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize data quality checker.
        
        Args:
            verbose: Whether to log detailed information
        """
        self.verbose = verbose
        logger.info("Data Quality Checker initialized")
    
    def check_data_leakage(
        self,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        feature_cols: List[str]
    ) -> Dict:
        """
        Detect various types of data leakage.
        
        Types checked:
        1. Duplicate rows between train/test
        2. Temporal leakage (test dates in train)
        3. Target leakage (features highly correlated with target)
        4. Look-ahead bias (future information in features)
        
        Args:
            train_data: Training dataset
            test_data: Test dataset
            feature_cols: List of feature column names
            
        Returns:
            Dictionary with leakage detection results
        """
        logger.info("🔍 Checking for data leakage...")
        issues = []
        
        # 1. Check for duplicate rows between train and test
        train_hashes = set(train_data[feature_cols].apply(tuple, axis=1))
        test_hashes = set(test_data[feature_cols].apply(tuple, axis=1))
        duplicates = train_hashes.intersection(test_hashes)
        
        if duplicates:
            issues.append({
                "type": "duplicate_rows",
                "severity": "HIGH",
                "count": len(duplicates),
                "message": f"{len(duplicates)} identical rows found in both train and test sets"
            })
            logger.warning(f"⚠️  Found {len(duplicates)} duplicate rows")
        
        # 2. Check for temporal leakage
        if 'date' in train_data.columns and 'date' in test_data.columns:
            train_data_sorted = train_data.copy()
            test_data_sorted = test_data.copy()
            
            if not pd.api.types.is_datetime64_any_dtype(train_data_sorted['date']):
                train_data_sorted['date'] = pd.to_datetime(train_data_sorted['date'])
                test_data_sorted['date'] = pd.to_datetime(test_data_sorted['date'])
            
            max_train_date = train_data_sorted['date'].max()
            min_test_date = test_data_sorted['date'].min()
            
            if min_test_date <= max_train_date:
                issues.append({
                    "type": "temporal_leakage",
                    "severity": "CRITICAL",
                    "message": f"Test data starts at {min_test_date}, but train data extends to {max_train_date}",
                    "max_train_date": str(max_train_date),
                    "min_test_date": str(min_test_date)
                })
                logger.error(f"❌ CRITICAL: Temporal leakage detected!")
        
        # 3. Check for target leakage (features too correlated with target)
        if 'target' in train_data.columns:
            correlations = train_data[feature_cols + ['target']].corr()['target'].abs()
            high_corr_features = correlations[correlations > 0.95].drop('target', errors='ignore').index.tolist()
            
            if high_corr_features:
                issues.append({
                    "type": "target_leakage",
                    "severity": "HIGH",
                    "features": high_corr_features,
                    "correlations": {feat: float(correlations[feat]) for feat in high_corr_features},
                    "message": f"Features suspiciously correlated with target (>0.95): {high_corr_features}"
                })
                logger.warning(f"⚠️  Potential target leakage: {high_corr_features}")
        
        # 4. Check for look-ahead bias (future information in feature names)
        future_keywords = ['future', 'next', 'forward', 't+1', 'lag-', 'lead']
        leaky_features = []
        
        for col in feature_cols:
            if any(keyword in col.lower() for keyword in future_keywords):
                leaky_features.append(col)
        
        if leaky_features:
            issues.append({
                "type": "look_ahead_bias",
                "severity": "MEDIUM",
                "features": leaky_features,
                "message": f"Features may contain future information: {leaky_features}"
            })
            logger.warning(f"⚠️  Possible look-ahead bias in features: {leaky_features}")
        
        # 5. Check for train/test contamination (test indices in train)
        if 'index' in train_data.columns and 'index' in test_data.columns:
            train_indices = set(train_data['index'])
            test_indices = set(test_data['index'])
            contamination = train_indices.intersection(test_indices)
            
            if contamination:
                issues.append({
                    "type": "index_contamination",
                    "severity": "CRITICAL",
                    "count": len(contamination),
                    "message": f"{len(contamination)} indices appear in both train and test"
                })
                logger.error(f"❌ Index contamination detected: {len(contamination)} rows")
        
        result = {
            "leakage_detected": len(issues) > 0,
            "total_issues": len(issues),
            "issues": issues,
            "timestamp": datetime.now().isoformat()
        }
        
        if not issues:
            logger.info("✅ No data leakage detected")
        
        return result
    
    def check_missing_values(
        self,
        data: pd.DataFrame,
        threshold_pct: float = 50.0
    ) -> Dict:
        """
        Check for missing values in the dataset.
        
        Args:
            data: Dataset to check
            threshold_pct: Threshold percentage for flagging columns
            
        Returns:
            Dictionary with missing value statistics
        """
        logger.info("🔍 Checking for missing values...")
        
        missing_counts = data.isnull().sum()
        missing_pct = (missing_counts / len(data)) * 100
        
        problematic_cols = missing_pct[missing_pct > threshold_pct].to_dict()
        
        result = {
            "has_missing": missing_counts.sum() > 0,
            "total_missing": int(missing_counts.sum()),
            "missing_by_column": {
                col: {"count": int(count), "percentage": float(missing_pct[col])}
                for col, count in missing_counts[missing_counts > 0].items()
            },
            "problematic_columns": problematic_cols,
            "threshold_pct": threshold_pct
        }
        
        if problematic_cols:
            logger.warning(f"⚠️  {len(problematic_cols)} columns exceed {threshold_pct}% missing")
        else:
            logger.info("✅ Missing values within acceptable range")
        
        return result
    
    def check_outliers(
        self,
        data: pd.DataFrame,
        method: str = "iqr",
        threshold: float = 3.0
    ) -> Dict:
        """
        Detect outliers using IQR or Z-score method.
        
        Args:
            data: Dataset to check
            method: 'iqr' or 'zscore'
            threshold: Threshold for Z-score method (default: 3.0)
            
        Returns:
            Dictionary with outlier statistics
        """
        logger.info(f"🔍 Detecting outliers using {method.upper()} method...")
        
        outliers = {}
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            col_data = data[col].dropna()
            
            if len(col_data) == 0:
                continue
            
            if method == "iqr":
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outlier_mask = (col_data < lower_bound) | (col_data > upper_bound)
            
            elif method == "zscore":
                z_scores = np.abs((col_data - col_data.mean()) / col_data.std())
                outlier_mask = z_scores > threshold
            
            else:
                raise ValueError(f"Unknown method: {method}")
            
            outlier_count = outlier_mask.sum()
            
            if outlier_count > 0:
                outliers[col] = {
                    "count": int(outlier_count),
                    "percentage": float(outlier_count / len(col_data) * 100),
                    "method": method
                }
        
        result = {
            "has_outliers": len(outliers) > 0,
            "outliers_by_column": outliers,
            "method": method,
            "threshold": threshold if method == "zscore" else None
        }
        
        if outliers:
            logger.warning(f"⚠️  Outliers detected in {len(outliers)} columns")
        else:
            logger.info("✅ No significant outliers detected")
        
        return result
    
    def check_distribution_shift(
        self,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        alpha: float = 0.05
    ) -> Dict:
        """
        Check for distribution shift using Kolmogorov-Smirnov test.
        
        Args:
            train_data: Training data
            test_data: Test data
            alpha: Significance level (default: 0.05)
            
        Returns:
            Dictionary with distribution shift results
        """
        logger.info("🔍 Checking for distribution shifts...")
        
        shifts = {}
        numeric_cols = train_data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col not in test_data.columns:
                continue
            
            train_col = train_data[col].dropna()
            test_col = test_data[col].dropna()
            
            if len(train_col) < 3 or len(test_col) < 3:
                continue
            
            # Kolmogorov-Smirnov test
            statistic, pvalue = stats.ks_2samp(train_col, test_col)
            
            shifted = pvalue < alpha
            
            shifts[col] = {
                "statistic": float(statistic),
                "pvalue": float(pvalue),
                "shifted": shifted,
                "severity": "HIGH" if pvalue < 0.01 else "MEDIUM" if shifted else "LOW"
            }
        
        shifted_features = {k: v for k, v in shifts.items() if v["shifted"]}
        
        result = {
            "distribution_shift_detected": len(shifted_features) > 0,
            "shifted_features": shifted_features,
            "all_tests": shifts,
            "alpha": alpha
        }
        
        if shifted_features:
            logger.warning(f"⚠️  Distribution shift detected in {len(shifted_features)} features")
        else:
            logger.info("✅ No significant distribution shifts detected")
        
        return result
    
    def check_class_imbalance(
        self,
        data: pd.DataFrame,
        target_col: str,
        threshold_ratio: float = 10.0
    ) -> Dict:
        """
        Check for class imbalance (for classification tasks).
        
        Args:
            data: Dataset with target column
            target_col: Name of target column
            threshold_ratio: Threshold for majority/minority ratio
            
        Returns:
            Dictionary with class imbalance information
        """
        logger.info("🔍 Checking for class imbalance...")
        
        if target_col not in data.columns:
            return {"error": f"Target column '{target_col}' not found"}
        
        class_counts = data[target_col].value_counts()
        class_percentages = (class_counts / len(data) * 100).to_dict()
        
        if len(class_counts) < 2:
            return {"error": "Only one class found"}
        
        majority_count = class_counts.max()
        minority_count = class_counts.min()
        imbalance_ratio = majority_count / minority_count
        
        result = {
            "imbalanced": imbalance_ratio > threshold_ratio,
            "imbalance_ratio": float(imbalance_ratio),
            "class_counts": class_counts.to_dict(),
            "class_percentages": class_percentages,
            "majority_class": str(class_counts.idxmax()),
            "minority_class": str(class_counts.idxmin()),
            "threshold_ratio": threshold_ratio
        }
        
        if result["imbalanced"]:
            logger.warning(
                f"⚠️  Class imbalance detected: ratio = {imbalance_ratio:.2f} "
                f"(threshold: {threshold_ratio})"
            )
        else:
            logger.info("✅ Classes are reasonably balanced")
        
        return result
    
    def run_all_checks(
        self,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        feature_cols: List[str],
        target_col: Optional[str] = None
    ) -> Dict:
        """
        Run all data quality checks.
        
        Args:
            train_data: Training dataset
            test_data: Test dataset
            feature_cols: Feature column names
            target_col: Target column name (optional)
            
        Returns:
            Comprehensive data quality report
        """
        logger.info("="*80)
        logger.info("RUNNING COMPREHENSIVE DATA QUALITY CHECKS")
        logger.info("="*80)
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "train_shape": train_data.shape,
            "test_shape": test_data.shape,
            "checks": {}
        }
        
        # 1. Data leakage
        results["checks"]["data_leakage"] = self.check_data_leakage(
            train_data, test_data, feature_cols
        )
        
        # 2. Missing values
        results["checks"]["missing_values_train"] = self.check_missing_values(train_data)
        results["checks"]["missing_values_test"] = self.check_missing_values(test_data)
        
        # 3. Outliers
        results["checks"]["outliers_train"] = self.check_outliers(train_data)
        results["checks"]["outliers_test"] = self.check_outliers(test_data)
        
        # 4. Distribution shift
        results["checks"]["distribution_shift"] = self.check_distribution_shift(
            train_data, test_data
        )
        
        # 5. Class imbalance (if classification)
        if target_col and target_col in train_data.columns:
            results["checks"]["class_imbalance"] = self.check_class_imbalance(
                train_data, target_col
            )
        
        # Aggregate critical issues
        critical_issues = []
        
        if results["checks"]["data_leakage"]["leakage_detected"]:
            for issue in results["checks"]["data_leakage"]["issues"]:
                if issue["severity"] == "CRITICAL":
                    critical_issues.append(issue)
        
        # Summary
        results["summary"] = {
            "critical_issues_count": len(critical_issues),
            "critical_issues": critical_issues,
            "passed": len(critical_issues) == 0,
            "warnings_count": (
                len(results["checks"]["data_leakage"]["issues"]) +
                (1 if results["checks"]["missing_values_train"]["has_missing"] else 0) +
                (1 if results["checks"]["outliers_train"]["has_outliers"] else 0) +
                (1 if results["checks"]["distribution_shift"]["distribution_shift_detected"] else 0)
            )
        }
        
        logger.info("="*80)
        logger.info("DATA QUALITY CHECK SUMMARY")
        logger.info("="*80)
        logger.info(f"Critical issues: {results['summary']['critical_issues_count']}")
        logger.info(f"Warnings: {results['summary']['warnings_count']}")
        logger.info(f"Overall status: {'✅ PASSED' if results['summary']['passed'] else '❌ FAILED'}")
        logger.info("="*80)
        
        return results

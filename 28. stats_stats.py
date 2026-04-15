"""
Statistical analysis module
Execution Order: 34
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
import warnings

logger = logging.getLogger(__name__)


class StatisticalValidator:
    """
    Statistical validation and hypothesis testing
    
    Implements:
    1. Bootstrap hypothesis testing
    2. Confidence interval estimation
    3. Effect size calculation
    4. Multiple comparison correction
    """
    
    def __init__(self, n_bootstrap: int = 10000, alpha: float = 0.05):
        """
        Initialize validator
        
        Args:
            n_bootstrap: Number of bootstrap samples
            alpha: Significance level
        """
        self.n_bootstrap = n_bootstrap
        self.alpha = alpha
        
        logger.info(f"Statistical validator initialized (α={alpha}, n_bootstrap={n_bootstrap})")
    
    def bootstrap_test(self, method1_scores: List[float], method2_scores: List[float],
                      alternative: str = 'two-sided') -> Dict[str, Any]:
        """
        Paired bootstrap hypothesis test
        
        Args:
            method1_scores: Scores from method 1
            method2_scores: Scores from method 2
            alternative: 'two-sided', 'greater', or 'less'
            
        Returns:
            Test results dictionary
        """
        n = len(method1_scores)
        if n != len(method2_scores):
            raise ValueError("Score lists must have same length")
        
        if n < 2:
            return self._insufficient_data_result()
        
        # Compute observed difference
        observed_diff = np.mean(method1_scores) - np.mean(method2_scores)
        
        # Bootstrap resampling
        bootstrap_diffs = self._bootstrap_resample(method1_scores, method2_scores)
        
        # Calculate p-value
        p_value = self._calculate_p_value(bootstrap_diffs, observed_diff, alternative)
        
        # Calculate confidence interval
        ci_lower, ci_upper = self._calculate_confidence_interval(bootstrap_diffs)
        
        # Calculate effect sizes
        effect_sizes = self._calculate_effect_sizes(method1_scores, method2_scores, observed_diff)
        
        # Test result
        significant = p_value < self.alpha
        
        result = {
            'p_value': p_value,
            'significant': significant,
            'observed_diff': observed_diff,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'n_samples': n,
            'alpha': self.alpha,
            'alternative': alternative,
            'effect_sizes': effect_sizes,
            'method1_mean': np.mean(method1_scores),
            'method1_std': np.std(method1_scores),
            'method2_mean': np.mean(method2_scores),
            'method2_std': np.std(method2_scores)
        }
        
        logger.info(f"Bootstrap test: p={p_value:.6f}, significant={significant}, "
                   f"diff={observed_diff:.4f}, CI=[{ci_lower:.4f}, {ci_upper:.4f}]")
        
        return result
    
    def _bootstrap_resample(self, scores1: List[float], scores2: List[float]) -> np.ndarray:
        """Perform bootstrap resampling"""
        n = len(scores1)
        bootstrap_diffs = np.zeros(self.n_bootstrap)
        
        for i in range(self.n_bootstrap):
            # Sample with replacement
            indices = np.random.choice(n, n, replace=True)
            
            # Compute bootstrap difference
            bootstrap_diff = (np.mean([scores1[idx] for idx in indices]) - 
                            np.mean([scores2[idx] for idx in indices]))
            bootstrap_diffs[i] = bootstrap_diff
        
        return bootstrap_diffs
    
    def _calculate_p_value(self, bootstrap_diffs: np.ndarray, observed_diff: float,
                          alternative: str) -> float:
        """Calculate bootstrap p-value"""
        if alternative == 'two-sided':
            p_value = np.mean(np.abs(bootstrap_diffs) >= np.abs(observed_diff))
        elif alternative == 'greater':
            p_value = np.mean(bootstrap_diffs >= observed_diff)
        elif alternative == 'less':
            p_value = np.mean(bootstrap_diffs <= observed_diff)
        else:
            raise ValueError(f"Unknown alternative: {alternative}")
        
        return float(p_value)
    
    def _calculate_confidence_interval(self, bootstrap_diffs: np.ndarray) -> Tuple[float, float]:
        """Calculate bootstrap confidence interval"""
        ci_lower = np.percentile(bootstrap_diffs, (self.alpha / 2) * 100)
        ci_upper = np.percentile(bootstrap_diffs, (1 - self.alpha / 2) * 100)
        
        return float(ci_lower), float(ci_upper)
    
    def _calculate_effect_sizes(self, scores1: List[float], scores2: List[float],
                               observed_diff: float) -> Dict[str, float]:
        """Calculate various effect sizes"""
        n1, n2 = len(scores1), len(scores2)
        
        # Cohen's d
        pooled_std = np.sqrt(((n1 - 1) * np.var(scores1) + (n2 - 1) * np.var(scores2)) / (n1 + n2 - 2))
        cohens_d = observed_diff / (pooled_std + 1e-10)
        
        # Hedges' g (corrected for small sample bias)
        correction = 1 - (3 / (4 * (n1 + n2) - 9))
        hedges_g = cohens_d * correction
        
        # Glass's delta (using control group SD)
        glass_delta = observed_diff / (np.std(scores2) + 1e-10)
        
        # Common language effect size (probability that random score from group1 > group2)
        cles = self._calculate_cles(scores1, scores2)
        
        return {
            'cohens_d': float(cohens_d),
            'hedges_g': float(hedges_g),
            'glass_delta': float(glass_delta),
            'cles': float(cles)
        }
    
    def _calculate_cles(self, scores1: List[float], scores2: List[float]) -> float:
        """Calculate Common Language Effect Size"""
        n1, n2 = len(scores1), len(scores2)
        count_greater = 0
        
        for x in scores1:
            for y in scores2:
                if x > y:
                    count_greater += 1
        
        return count_greater / (n1 * n2)
    
    def _insufficient_data_result(self) -> Dict[str, Any]:
        """Return result for insufficient data"""
        return {
            'p_value': 1.0,
            'significant': False,
            'observed_diff': 0.0,
            'ci_lower': 0.0,
            'ci_upper': 0.0,
            'n_samples': 0,
            'alpha': self.alpha,
            'alternative': 'two-sided',
            'effect_sizes': {
                'cohens_d': 0.0,
                'hedges_g': 0.0,
                'glass_delta': 0.0,
                'cles': 0.5
            },
            'method1_mean': 0.0,
            'method1_std': 0.0,
            'method2_mean': 0.0,
            'method2_std': 0.0,
            'warning': 'Insufficient data for statistical test'
        }
    
    def welch_t_test(self, method1_scores: List[float], method2_scores: List[float],
                    alternative: str = 'two-sided') -> Dict[str, Any]:
        """
        Welch's t-test (unequal variances)
        
        Args:
            method1_scores: Scores from method 1
            method2_scores: Scores from method 2
            alternative: 'two-sided', 'greater', or 'less'
            
        Returns:
            Test results dictionary
        """
        n1, n2 = len(method1_scores), len(method2_scores)
        
        if n1 < 2 or n2 < 2:
            return self._insufficient_data_result()
        
        # Perform Welch's t-test
        t_stat, p_value = stats.ttest_ind(method1_scores, method2_scores, 
                                         equal_var=False, 
                                         alternative=alternative)
        
        # Calculate means and standard deviations
        mean1, std1 = np.mean(method1_scores), np.std(method1_scores)
        mean2, std2 = np.mean(method2_scores), np.std(method2_scores)
        
        # Calculate effect sizes
        observed_diff = mean1 - mean2
        effect_sizes = self._calculate_effect_sizes(method1_scores, method2_scores, observed_diff)
        
        # Calculate confidence interval for difference
        se = np.sqrt(std1**2/n1 + std2**2/n2)
        df = (std1**2/n1 + std2**2/n2)**2 / ((std1**2/n1)**2/(n1-1) + (std2**2/n2)**2/(n2-1))
        
        if alternative == 'two-sided':
            t_critical = stats.t.ppf(1 - self.alpha/2, df)
            ci_lower = observed_diff - t_critical * se
            ci_upper = observed_diff + t_critical * se
        elif alternative == 'greater':
            t_critical = stats.t.ppf(1 - self.alpha, df)
            ci_lower = observed_diff - t_critical * se
            ci_upper = np.inf
        else:  # 'less'
            t_critical = stats.t.ppf(1 - self.alpha, df)
            ci_lower = -np.inf
            ci_upper = observed_diff + t_critical * se
        
        result = {
            'test': 'welch_t_test',
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'significant': p_value < self.alpha,
            'observed_diff': float(observed_diff),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            'degrees_freedom': float(df),
            'n1': n1,
            'n2': n2,
            'mean1': float(mean1),
            'std1': float(std1),
            'mean2': float(mean2),
            'std2': float(std2),
            'effect_sizes': effect_sizes,
            'alpha': self.alpha,
            'alternative': alternative
        }
        
        logger.info(f"Welch's t-test: t={t_stat:.4f}, p={p_value:.6f}, "
                   f"significant={p_value < self.alpha}")
        
        return result
    
    def mann_whitney_u_test(self, method1_scores: List[float], method2_scores: List[float],
                           alternative: str = 'two-sided') -> Dict[str, Any]:
        """
        Mann-Whitney U test (non-parametric)
        
        Args:
            method1_scores: Scores from method 1
            method2_scores: Scores from method 2
            alternative: 'two-sided', 'greater', or 'less'
            
        Returns:
            Test results dictionary
        """
        n1, n2 = len(method1_scores), len(method2_scores)
        
        if n1 < 2 or n2 < 2:
            return self._insufficient_data_result()
        
        # Perform Mann-Whitney U test
        u_stat, p_value = stats.mannwhitneyu(method1_scores, method2_scores, 
                                            alternative=alternative)
        
        # Calculate ranks
        all_scores = method1_scores + method2_scores
        ranks = stats.rankdata(all_scores)
        
        # Calculate effect size (rank-biserial correlation)
        r1 = np.mean(ranks[:n1])
        r2 = np.mean(ranks[n1:])
        rank_biserial = 2 * (r1 - r2) / (n1 + n2)
        
        # Calculate medians and IQRs
        median1, iqr1 = np.median(method1_scores), stats.iqr(method1_scores)
        median2, iqr2 = np.median(method2_scores), stats.iqr(method2_scores)
        
        result = {
            'test': 'mann_whitney_u_test',
            'u_statistic': float(u_stat),
            'p_value': float(p_value),
            'significant': p_value < self.alpha,
            'median1': float(median1),
            'iqr1': float(iqr1),
            'median2': float(median2),
            'iqr2': float(iqr2),
            'rank_biserial': float(rank_biserial),
            'n1': n1,
            'n2': n2,
            'alpha': self.alpha,
            'alternative': alternative
        }
        
        logger.info(f"Mann-Whitney U test: U={u_stat:.4f}, p={p_value:.6f}, "
                   f"significant={p_value < self.alpha}")
        
        return result
    
    def multiple_comparison_correction(self, p_values: List[float],
                                      method: str = 'holm') -> Dict[str, Any]:
        """
        Apply multiple comparison correction
        
        Args:
            p_values: List of p-values
            method: Correction method ('holm', 'bonferroni', 'fdr_bh')
            
        Returns:
            Correction results
        """
        if not p_values:
            return {'corrected_p_values': [], 'significant': []}
        
        p_array = np.array(p_values)
        
        if method == 'holm':
            from statsmodels.stats.multitest import multipletests
            reject, corrected_p, _, _ = multipletests(p_array, alpha=self.alpha, 
                                                     method='holm')
        elif method == 'bonferroni':
            corrected_p = p_array * len(p_array)
            corrected_p = np.minimum(corrected_p, 1.0)
            reject = corrected_p < self.alpha
        elif method == 'fdr_bh':
            from statsmodels.stats.multitest import multipletests
            reject, corrected_p, _, _ = multipletests(p_array, alpha=self.alpha,
                                                     method='fdr_bh')
        else:
            raise ValueError(f"Unknown correction method: {method}")
        
        result = {
            'method': method,
            'original_p_values': p_values,
            'corrected_p_values': corrected_p.tolist(),
            'significant': reject.tolist(),
            'alpha': self.alpha,
            'n_tests': len(p_values)
        }
        
        logger.info(f"Multiple comparison correction ({method}): "
                   f"{sum(reject)}/{len(p_values)} significant after correction")
        
        return result
    
    def power_analysis(self, effect_size: float, alpha: float = None,
                      power: float = 0.8, ratio: float = 1.0) -> Dict[str, Any]:
        """
        Power analysis for sample size determination
        
        Args:
            effect_size: Cohen's d effect size
            alpha: Significance level (default: self.alpha)
            power: Desired power (1 - beta)
            ratio: n2/n1 ratio
            
        Returns:
            Power analysis results
        """
        if alpha is None:
            alpha = self.alpha
        
        from statsmodels.stats.power import TTestIndPower
        
        power_analysis = TTestIndPower()
        
        # Calculate required sample size
        n_per_group = power_analysis.solve_power(
            effect_size=effect_size,
            alpha=alpha,
            power=power,
            ratio=ratio
        )
        
        # Round up to nearest integer
        n_per_group = int(np.ceil(n_per_group))
        
        result = {
            'effect_size': effect_size,
            'alpha': alpha,
            'power': power,
            'ratio': ratio,
            'required_n_per_group': n_per_group,
            'required_total_n': int(n_per_group * (1 + ratio))
        }
        
        logger.info(f"Power analysis: effect_size={effect_size:.2f}, alpha={alpha}, "
                   f"power={power} -> n_per_group={n_per_group}")
        
        return result
    
    def calculate_bayesian_factor(self, method1_scores: List[float],
                                method2_scores: List[float]) -> Dict[str, Any]:
        """
        Calculate Bayesian factor for hypothesis testing
        
        Args:
            method1_scores: Scores from method 1
            method2_scores: Scores from method 2
            
        Returns:
            Bayesian analysis results
        """
        try:
            from bayesian_optimization import BayesFactor
            
            # Simplified Bayesian factor calculation
            # In practice, use a proper Bayesian model
            
            n1, n2 = len(method1_scores), len(method2_scores)
            
            if n1 < 2 or n2 < 2:
                return {'bayes_factor': 1.0, 'evidence': 'Insufficient data'}
            
            # Calculate means and pooled variance
            mean1, var1 = np.mean(method1_scores), np.var(method1_scores)
            mean2, var2 = np.mean(method2_scores), np.var(method2_scores)
            pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
            
            # Simplified Bayes factor (Kass & Raftery, 1995)
            t_stat = (mean1 - mean2) / np.sqrt(pooled_var * (1/n1 + 1/n2))
            df = n1 + n2 - 2
            
            # Bayes factor approximation
            bayes_factor = np.sqrt((n1 + n2) / (n1 * n2)) * \
                          (1 + t_stat**2 / df) ** (-(df + 1) / 2)
            
            # Interpret evidence
            if bayes_factor > 100:
                evidence = "Extreme evidence for H1"
            elif bayes_factor > 30:
                evidence = "Very strong evidence for H1"
            elif bayes_factor > 10:
                evidence = "Strong evidence for H1"
            elif bayes_factor > 3:
                evidence = "Moderate evidence for H1"
            elif bayes_factor > 1:
                evidence = "Anecdotal evidence for H1"
            elif bayes_factor > 1/3:
                evidence = "Anecdotal evidence for H0"
            elif bayes_factor > 1/10:
                evidence = "Moderate evidence for H0"
            elif bayes_factor > 1/30:
                evidence = "Strong evidence for H0"
            elif bayes_factor > 1/100:
                evidence = "Very strong evidence for H0"
            else:
                evidence = "Extreme evidence for H0"
            
            result = {
                'bayes_factor': float(bayes_factor),
                'evidence': evidence,
                'log_bayes_factor': float(np.log(bayes_factor)),
                't_statistic': float(t_stat),
                'degrees_freedom': int(df),
                'n1': n1,
                'n2': n2
            }
            
            logger.info(f"Bayesian factor: BF={bayes_factor:.4f}, evidence={evidence}")
            
            return result
            
        except ImportError:
            logger.warning("Bayesian optimization package not available")
            return {'bayes_factor': 1.0, 'evidence': 'Calculation not available'}

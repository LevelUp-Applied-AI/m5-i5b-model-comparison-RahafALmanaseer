# Tree vs. Linear Disagreement Analysis

## Sample Details

- **Test-set index:** 4060
- **True label:** 0
- **RF predicted P(churn=1):** 0.5998
- **LR predicted P(churn=1):** 0.1700
- **Probability difference:** 0.4299

## Feature Values# Tree vs. Linear Disagreement Analysis

## Sample Details

- **Test-set index:** 4060
- **True label:** 0
- **RF predicted P(churn=1):** 0.5998
- **LR predicted P(churn=1):** 0.1700
- **Probability difference:** 0.4299

## Feature Values

- **tenure:** 36.0
- **monthly_charges:** 20.0
- **total_charges:** 1077.33
- **num_support_calls:** 2.0
- **senior_citizen:** 0.0
- **has_partner:** 0.0
- **has_dependents:** 0.0
- **contract_months:** 1.0

## Structural Explanation

The models disagree on this sample because the Random Forest captured a critical interaction between the short contract duration (1 month) and the tenure, recognizing that month-to-month contracts represent a high-risk churn signal. While the Logistic Regression model assumes a simple linear benefit from the 36-month tenure, the tree-based model identifies a threshold effect where customer loyalty (tenure) is overshadowed by the volatility of a monthly contract.

- **tenure:** 36.0
- **monthly_charges:** 20.0
- **total_charges:** 1077.33
- **num_support_calls:** 2.0
- **senior_citizen:** 0.0
- **has_partner:** 0.0
- **has_dependents:** 0.0
- **contract_months:** 1.0

## Structural Explanation

The models disagree because the Random Forest identified a high-risk interaction between the month-to-month contract and the tenure. While the Logistic Regression model sees the 36-month tenure as a simple linear factor that reduces churn risk, the tree-based model captures a threshold effect where even long-term customers become highly likely to churn if they are on a flexible, one-month contract.
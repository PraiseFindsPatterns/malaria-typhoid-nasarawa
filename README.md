# Malaria and Typhoid Cases at General Hospital Nasarawa (2005-2022)
Analysis of 18 years of annual hospital case counts. Originally done in SPSS and Minitab for my National Diploma in Statistics, then re-implemented in Python.
## Key Findings
- **Malaria:** mean 768 cases/year (range 527 to 994). A linear trend of +9.9 cases/year fits poorly (R² = 0.12), so there is no clear long-term trend.
- **Typhoid:** mean 876 cases/year (range 541 to 1,337). A trend of -8.4 cases/year also fits poorly (R² = 0.05).
- **Relationship:** correlation between the two diseases is -0.09, essentially none.
- **Forecast:** the extrapolation to 2042 is illustrative only, not a prediction.
## Limitations
- One hospital, 18 annual counts. These are hospital cases, not community incidence.
- Year-to-year variation is large relative to any trend. Possible explanations (attendance, diagnosis practices, population change) could not be tested with this data.
## Run it    
pip install -r requirements.txt
python original_work/analysis.py
## Author
Onu Veronica Praise
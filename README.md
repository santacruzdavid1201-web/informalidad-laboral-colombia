# Informal Labor in Colombia: A Multilevel Econometric and Data Analytics Approach

![Python](https://img.shields.io/badge/Python-3.12-blue)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-336791)
![Power%20BI](https://img.shields.io/badge/Power_BI-Dashboard-F2C811)
![Econometrics](https://img.shields.io/badge/Econometrics-Panel%20%2B%20Heckman-success)

This project investigates the determinants of labor informality in Colombia by integrating macroeconomic panel data with microeconometric analysis. The workflow combines data engineering, econometric modeling, SQL, Python, and interactive dashboards to generate reproducible evidence for policy analysis.

---

# Context

Labor informality remains one of Colombia's most persistent structural challenges, affecting productivity, tax revenues, social protection, and regional development.

Rather than studying the phenomenon from a single perspective, this project combines structural regional indicators with individual labor market characteristics to provide a more comprehensive understanding of the drivers of informality.

---

# Methodology

The project integrates two complementary analytical approaches:

* **Macroeconomic analysis:** Department-level panel data models evaluating how competitiveness dimensions relate to labor informality.
* **Microeconomic analysis:** Heckman selection models using household survey data to estimate individual determinants while correcting for sample selection bias.

The complete theoretical framework, econometric specification, robustness tests, and policy discussion are available in the accompanying academic thesis.

---

# Tech Stack

* Python
* SQL (PostgreSQL)
* Power BI
* Pandas
* NumPy
* Statsmodels
* Linearmodels
* Scikit-learn
* Git & GitHub

---

# Data Pipeline

```text
Raw CSV files
      │
      ▼
Python
(Cleaning • ETL • Feature Engineering)
      │
      ▼
PostgreSQL
      │
      ▼
Econometric Models
      │
      ▼
Power BI Dashboard
```

---

# Key Findings

* Higher education (compared with primary education) reduces the estimated probability of labor informality by **24–28 percentage points**, even after correcting for sample selection using a Heckman model.
* A four-factor competitiveness framework—covering **digital human capital, territorial infrastructure, labor conditions, and innovation**—explains regional differences in informality, with statistically significant effects in the department-level Random Effects specification (validated through Hausman and Mundlak tests).
* Replacing childcare variables with **care responsibilities for elderly, sick, or disabled household members** produced a substantially stronger exclusion restriction, reducing the correlation between the inverse Mills ratio and the exclusion variable from approximately **0.71 to 0.055**, improving identification of the Heckman model.

---

# Results

> *(Replace the placeholders below with screenshots from the repository.)*

### Interactive Power BI Dashboard

*Dashboard overview*

![Dashboard](images/dashboard_overview.png)

### Regional Informality Map

*Department-level visualization of labor informality.*

![Map](images/informality_map.png)

### Econometric Results

*Interactive visualization of macroeconomic determinants.*

![Results](images/model_results.png)

---

# Repository Structure

```text
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── sql/
├── src/
├── dashboard/
├── figures/
├── docs/
└── README.md
```

---

# Reproducing the Project

1. Clone the repository.
2. Install the required Python dependencies.
3. Import the processed datasets (or generate them from the raw data).
4. Execute the ETL pipeline.
5. Estimate the econometric models.
6. Connect PostgreSQL to Power BI to explore the interactive dashboard.

---

# Author's Note

This repository is a public and simplified version of my undergraduate Economics thesis. It emphasizes the programming workflow, reproducible data pipeline, and analytical implementation while intentionally omitting the complete academic manuscript, literature review, and methodological discussion to preserve academic integrity.

---

# Contact

**David Santacruz**

* LinkedIn: *(add your profile)*
* GitHub: *(add your profile)*

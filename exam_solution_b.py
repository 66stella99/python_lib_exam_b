import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
pd.set_option("display.max_columns", None)

jobs = pd.read_csv("jobs.csv")
employees = pd.read_csv("employees.csv")
managers = pd.read_csv("managers.csv")
positions = pd.read_csv("positions.csv")

for d in [jobs, employees, managers, positions]:
    d.columns = d.columns.str.strip()

id_map = {}
for t in jobs.loc[jobs.duplicated(subset="job_title", keep=False), "job_title"].unique():
    rows = jobs[jobs["job_title"] == t].sort_values("job_id")
    keep_id = rows.iloc[0]["job_id"]
    for jid in rows.iloc[1:]["job_id"]:
        id_map[jid] = keep_id
jobs = jobs[~jobs["job_id"].isin(id_map.keys())].reset_index(drop=True)
positions["job_id"] = positions["job_id"].replace(id_map)

employees["manager_id"] = employees["manager_id"].fillna(-1).astype(int)
employees["age"] = employees["age"].fillna(employees["age"].median())
employees["gender"] = employees["gender"].fillna("Unknown")
employees["years_of_experience"] = employees["years_of_experience"].fillna(employees["years_of_experience"].median())
missing_salary_ids = employees.loc[employees["salary"].isna(), "employee_id"]
employees = employees.dropna(subset=["salary"])
positions = positions[positions["employee_id"].isin(employees["employee_id"])]

managers["age"] = managers["age"].fillna(managers["age"].median())
managers["gender"] = managers["gender"].fillna("Unknown")
managers["starting_date"] = pd.to_datetime(managers["starting_date"])
managers["starting_date"] = managers["starting_date"].fillna(managers["starting_date"].median())
missing_exp_ids = managers.loc[managers["years_of_experience"].isna(), "manager_id"]
managers = managers.dropna(subset=["years_of_experience"])
employees.loc[employees["manager_id"].isin(missing_exp_ids), "manager_id"] = -1
managers["years_of_experience"] = managers["years_of_experience"].astype(int)

positions = positions.dropna(subset=["job_id"])
positions["job_id"] = positions["job_id"].astype(int)
positions["starting_date"] = pd.to_datetime(positions["starting_date"])
positions["starting_date"] = positions["starting_date"].fillna(positions["starting_date"].median())
positions["performance_score"] = positions["performance_score"].fillna(positions["performance_score"].median()).astype(int)
# 1a
plt.figure(figsize=(16, 10))
nationality_counts = employees["nationality"].value_counts().reset_index()
nationality_counts.columns = ["nationality", "employee_count"]
plt.bar(nationality_counts["nationality"], nationality_counts["employee_count"])
plt.title("Employees by Nationality")
plt.show()
# 1b
plt.figure(figsize=(16, 10))

plt.hist(employees["age"], bins=15)
plt.title("Employee Age Distribution")
plt.show()
# 1c
cols = ["employee_id", "first_name", "last_name", "age", "years_of_experience", "salary"]
sorted_by_salary = employees.sort_values("salary")
bottom3_salary = sorted_by_salary[cols].head(3)
top3_salary = sorted_by_salary[cols].tail(3)
# 1d
plt.figure(figsize=(16, 10))
experience_salary_corr = employees["years_of_experience"].corr(employees["salary"])
plt.scatter(employees["years_of_experience"], employees["salary"])
plt.title(f"Experience vs Salary (corr={experience_salary_corr})")
plt.show()

# 2a
def perf_label(s):
    if s <= 5: return "Low Performance"
    elif s <= 7: return "Medium Performance"
    else: return "High Performance"

plt.figure(figsize=(16, 10))
managers["performance_label"] = managers["performance_score"].apply(perf_label)
performance_counts = managers.groupby("performance_label").size().reset_index(name="count")
plt.bar(performance_counts["performance_label"], performance_counts["count"])
plt.title("Managers by Performance Category")
plt.show()

# 2b
plt.figure(figsize=(16, 10))
managers["start_year"] = managers["starting_date"].dt.year
yearly_hires = managers.groupby("start_year").size()
plt.plot(yearly_hires.index, yearly_hires.values)
plt.title("Manager Hiring")
plt.show()

# 2c
plt.figure(figsize=(16, 10))
mean_salary_by_nationality = managers.groupby("nationality")["salary"].mean().sort_values(ascending=False).round(0)
plt.bar(mean_salary_by_nationality.index, mean_salary_by_nationality.values)
plt.xticks(rotation=75, ha="right")
plt.title("Mean Manager Salary by Nationality")
plt.show()

# 2d
plt.figure(figsize=(16, 10))
recent_managers = managers[managers["starting_date"] >= "2022-01-01"]
employees_per_manager = employees[employees["manager_id"] != -1].groupby("manager_id").size()
mean_team_size_recent_managers = recent_managers["manager_id"].map(employees_per_manager).fillna(0).mean()
print("2d - mean team size for managers hired since 2022-01-01:", mean_team_size_recent_managers)

# 2e
plt.figure(figsize=(16, 10))
managers["num_employees"] = managers["manager_id"].map(employees_per_manager).fillna(0)
team_size_salary_corr = managers["num_employees"].corr(managers["salary"])
plt.scatter(managers["num_employees"], managers["salary"])
plt.title(f"Team Size vs Manager Salary (corr={team_size_salary_corr})")
plt.show()

# 3a
plt.figure(figsize=(16, 10))
department_job_counts = jobs["department"].value_counts().reset_index()
department_job_counts.columns = ["department", "job_count"]
plt.bar(department_job_counts["department"], department_job_counts["job_count"])
plt.title("Jobs by Department")
plt.show()

# 4a
plt.figure(figsize=(16, 10))
positions_per_employee = positions.groupby("employee_id").size().reindex(employees["employee_id"], fill_value=0)
position_count_distribution = positions_per_employee.value_counts().sort_index()
plt.bar(position_count_distribution.index.astype(str), position_count_distribution.values)
plt.title("Employees by Number of Positions")
plt.show()

# 4b
employee_positions = positions.merge(employees[["employee_id", "nationality"]], on="employee_id", how="left")
employees_working_abroad = employee_positions.loc[
    employee_positions["nationality"] != employee_positions["position_location"], "employee_id"
].unique()
print("4b - employees working outside their nationality:", len(employees_working_abroad))

# 4c
mean_performance_by_employee = positions.groupby("employee_id")["performance_score"].mean()
employees["mean_performance_score"] = employees["employee_id"].map(mean_performance_by_employee)

# 4d
plt.figure(figsize=(16, 10))
emp_with_manager_perf = employees.dropna(subset=["mean_performance_score"])
emp_with_manager_perf = emp_with_manager_perf[emp_with_manager_perf["manager_id"] != -1]
emp_with_manager_perf = emp_with_manager_perf.merge(
    managers[["manager_id", "performance_score"]], on="manager_id", how="left"
).dropna(subset=["performance_score"])
employee_perf_manager_perf_corr = emp_with_manager_perf["mean_performance_score"].corr(emp_with_manager_perf["performance_score"])
plt.scatter(emp_with_manager_perf["performance_score"], emp_with_manager_perf["mean_performance_score"])
plt.title(f"Employee vs Manager Performance (corr={employee_perf_manager_perf_corr})")
plt.show()

# 4e
positions_with_dept = positions.merge(jobs[["job_id", "department"]], on="job_id", how="left")
employee_departments = positions_with_dept.drop_duplicates(subset=["employee_id", "department"])[["employee_id", "department"]]
department_salaries = employee_departments.merge(employees[["employee_id", "salary"]], on="employee_id", how="left")
department_salary_summary = department_salaries.groupby("department").agg(
    amount_of_employees=("employee_id", "nunique"),
    total_salary_amount=("salary", "sum")
).reset_index().sort_values("total_salary_amount", ascending=False)
top3_departments = department_salary_summary.head(3)
print("4e - top 3 departments by total salary:")
print(top3_departments)
# 4f
plt.figure(figsize=(16, 10))
emp_with_manager = employees[employees["manager_id"] != -1].merge(
    managers[["manager_id", "performance_score"]], on="manager_id", how="left"
).dropna(subset=["performance_score", "years_of_experience"])
experience_manager_perf_corr = emp_with_manager["years_of_experience"].corr(emp_with_manager["performance_score"])
plt.scatter(emp_with_manager["performance_score"], emp_with_manager["years_of_experience"])
plt.title(f"Employee Experience vs Manager Performance (corr={experience_manager_perf_corr})")
plt.show()

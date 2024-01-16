import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from scipy.stats import pearsonr


# Reload the CSV file due to code execution state reset
file_path = '/Users/junrunchen/Downloads/data_2023-Dec-14 (5).csv'
data = pd.read_csv(file_path)

# Descriptive statistical analysis of the original data
descriptive_stats = data[['newCasesBySpecimenDateRollingSum', 
                          'newCasesBySpecimenDateRollingRate', 
                          'newCasesBySpecimenDateChange', 
                          'newCasesBySpecimenDateChangePercentage']].describe()

# Including additional statistics like median and range
descriptive_stats.loc['median'] = data[['newCasesBySpecimenDateRollingSum', 
                                        'newCasesBySpecimenDateRollingRate', 
                                        'newCasesBySpecimenDateChange', 
                                        'newCasesBySpecimenDateChangePercentage']].median()
descriptive_stats.loc['range'] = data[['newCasesBySpecimenDateRollingSum', 
                                       'newCasesBySpecimenDateRollingRate', 
                                       'newCasesBySpecimenDateChange', 
                                       'newCasesBySpecimenDateChangePercentage']].max() - \
                                 data[['newCasesBySpecimenDateRollingSum', 
                                       'newCasesBySpecimenDateRollingRate', 
                                       'newCasesBySpecimenDateChange', 
                                       'newCasesBySpecimenDateChangePercentage']].min()

print(descriptive_stats)

# Load the hospitalization data files
hospitalization_file_path_1 = '/Users/junrunchen/Downloads/data_2023-Dec-14 (6).csv'
hospitalization_file_path_2 = '/Users/junrunchen/Downloads/data_2023-Dec-14 (7).csv'
public_health_measures_file_path = '/Users/junrunchen/Downloads/restrictions_weekly.csv'

hospitalization_data_1 = pd.read_csv(hospitalization_file_path_1)
hospitalization_data_2 = pd.read_csv(hospitalization_file_path_2)
public_health_measures_data = pd.read_csv(public_health_measures_file_path)

# Display the first few rows of each dataset to understand their structures
hospitalization_data_1.head(), hospitalization_data_2.head(), public_health_measures_data.head()

# Convert 'date' in hospitalization data to datetime objects for merging
hospitalization_data_1['date'] = pd.to_datetime(hospitalization_data_1['date'])
hospitalization_data_2['date'] = pd.to_datetime(hospitalization_data_2['date'])

# Convert 'date' in public health measures data to datetime objects for merging
public_health_measures_data['date'] = pd.to_datetime(public_health_measures_data['date'])

# Merging hospitalization data with public health measures data
# We use a left join to keep all hospitalization records and associate them with the relevant public health measures based on the week

# First, we create a 'date' column in hospitalization data to match with public health measures data
hospitalization_data_1['date'] = hospitalization_data_1['date'] - pd.to_timedelta(hospitalization_data_1['date'].dt.dayofweek, unit='d')
hospitalization_data_2['date'] = hospitalization_data_2['date'] - pd.to_timedelta(hospitalization_data_2['date'].dt.dayofweek, unit='d')

# Merging the datasets
merged_data_1 = hospitalization_data_1.merge(public_health_measures_data, on='date', how='left')
merged_data_2 = hospitalization_data_2.merge(public_health_measures_data, on='date', how='left')

merged_data_1.head(), merged_data_2.head()


# Selecting a public health measure for analysis, e.g., stay-at-home order
measure = 'stay_at_home'

# Splitting the data into two groups based on the selected public health measure
group_with_measure = merged_data_1[merged_data_1[measure] == 1]['newAdmissions']
group_without_measure = merged_data_1[merged_data_1[measure] == 0]['newAdmissions']

# Performing a t-test to compare the means of the two groups
t_stat, p_value = ttest_ind(group_with_measure, group_without_measure, equal_var=False, nan_policy='omit')

print(t_stat, p_value)


# Load the testing data files
testing_data_file_path_1 = '/Users/junrunchen/Downloads/data_2023-Dec-14 (8).csv'
testing_data_file_path_2 = '/Users/junrunchen/Downloads/data_2023-Dec-14 (9).csv'

testing_data_1 = pd.read_csv(testing_data_file_path_1)
testing_data_2 = pd.read_csv(testing_data_file_path_2)

# Display the first few rows of each dataset to understand their structures
testing_data_1.head(), testing_data_2.head()

# Convert 'date' in testing data to datetime objects for merging
testing_data_1['date'] = pd.to_datetime(testing_data_1['date'])
testing_data_2['date'] = pd.to_datetime(testing_data_2['date'])

# Merging the testing datasets on common columns
merged_testing_data = pd.merge(testing_data_1, testing_data_2, 
                               on=['areaType', 'areaName', 'areaCode', 'date'], 
                               how='inner')

# Display the first few rows of the merged dataset to verify the merge
merged_testing_data.head()


# Checking for NaNs and handling them
merged_testing_data_clean = merged_testing_data.copy()
merged_testing_data_clean = merged_testing_data_clean.dropna(subset=['uniqueCasePositivityBySpecimenDateRollingSum',
                                                                     'uniquePeopleTestedBySpecimenDateRollingSum',
                                                                     'newVirusTestsBySpecimenDate'])

# Recalculating correlation between positivity rates and the number of people tested
corr_positivity_vs_people_tested, p_value_positivity_vs_people_tested = pearsonr(
    merged_testing_data_clean['uniqueCasePositivityBySpecimenDateRollingSum'],
    merged_testing_data_clean['uniquePeopleTestedBySpecimenDateRollingSum']
)

# Recalculating correlation between positivity rates and new virus tests
corr_positivity_vs_new_tests, p_value_positivity_vs_new_tests = pearsonr(
    merged_testing_data_clean['uniqueCasePositivityBySpecimenDateRollingSum'],
    merged_testing_data_clean['newVirusTestsBySpecimenDate']
)

# Displaying the correlation coefficients and p-values
corr_results = {
    'Correlation (Positivity Rate vs People Tested)': corr_positivity_vs_people_tested,
    'P-Value (Positivity Rate vs People Tested)': p_value_positivity_vs_people_tested,
    'Correlation (Positivity Rate vs New Tests)': corr_positivity_vs_new_tests,
    'P-Value (Positivity Rate vs New Tests)': p_value_positivity_vs_new_tests
}

print(corr_results)

# Load the death data file
death_data_file_path = '/Users/junrunchen/Downloads/data_2023-Dec-14 (14).csv'

death_data = pd.read_csv(death_data_file_path)

# Display the first few rows of the death data to understand its structure
death_data.head()


# Load the case data file again
case_data_file_path = '/Users/junrunchen/Downloads/data_2023-Dec-14 (5).csv'
case_data = pd.read_csv(case_data_file_path)

# Convert 'date' to datetime for time-series analysis
case_data['date'] = pd.to_datetime(case_data['date'])

# Filter data for London
london_case_data = case_data[case_data['areaName'] == 'London']

# Time-series plot of COVID-19 case rates in London
plt.figure(figsize=(15, 6))
plt.plot(london_case_data['date'], london_case_data['newCasesBySpecimenDateRollingRate'], label='7-Day Rolling Rate of New Cases')
plt.title('COVID-19 Case Rates in London (7-Day Rolling Rate)')
plt.xlabel('Date')
plt.ylabel('Case Rate per 100,000')
plt.legend()
plt.grid(True)
plt.show()

# Perform descriptive statistical analysis
descriptive_statistics = london_case_data['newCasesBySpecimenDateRollingRate'].describe()
print(descriptive_statistics)

# Load the public health measures data file again
public_health_measures_file_path = '/Users/junrunchen/Downloads/restrictions_weekly.csv'
public_health_measures_data = pd.read_csv(public_health_measures_file_path)

# Convert 'date' in public health measures data to datetime objects for merging
public_health_measures_data['date'] = pd.to_datetime(public_health_measures_data['date'])

# Creating a 'date' column in case data to match with public health measures data
london_case_data['date'] = london_case_data['date'] - pd.to_timedelta(london_case_data['date'].dt.dayofweek, unit='d')

# Merging the case data with public health measures data
merged_case_measures_data = london_case_data.merge(public_health_measures_data, on='date', how='left')

# Performing hypothesis testing to assess the impact of school closures on COVID-19 transmission rates
# Splitting the data into two groups based on the school closures
group_with_school_closure = merged_case_measures_data[merged_case_measures_data['schools_closed'] == 1]['newCasesBySpecimenDateRollingRate']
group_without_school_closure = merged_case_measures_data[merged_case_measures_data['schools_closed'] == 0]['newCasesBySpecimenDateRollingRate']

# Performing a t-test to compare the means of the two groups
t_stat_school_closure, p_value_school_closure = ttest_ind(group_with_school_closure, group_without_school_closure, equal_var=False, nan_policy='omit')

# Performing hypothesis testing to assess the impact of social distancing measures on COVID-19 transmission rates
# Assuming 'stay_at_home' represents social distancing measures
group_with_social_distancing = merged_case_measures_data[merged_case_measures_data['stay_at_home'] == 1]['newCasesBySpecimenDateRollingRate']
group_without_social_distancing = merged_case_measures_data[merged_case_measures_data['stay_at_home'] == 0]['newCasesBySpecimenDateRollingRate']

# Performing a t-test to compare the means of the two groups
t_stat_social_distancing, p_value_social_distancing = ttest_ind(group_with_social_distancing, group_without_social_distancing, equal_var=False, nan_policy='omit')

print(t_stat_school_closure, p_value_school_closure, t_stat_social_distancing, p_value_social_distancing)

# Checking the distribution of data in the groups for school closures
school_closure_data_summary = {
    'Count with School Closure': group_with_school_closure.count(),
    'Mean with School Closure': group_with_school_closure.mean(),
    'Count without School Closure': group_without_school_closure.count(),
    'Mean without School Closure': group_without_school_closure.mean()
}

# Checking the distribution of data in the groups for social distancing measures
social_distancing_data_summary = {
    'Count with Social Distancing': group_with_social_distancing.count(),
    'Mean with Social Distancing': group_with_social_distancing.mean(),
    'Count without Social Distancing': group_without_social_distancing.count(),
    'Mean without Social Distancing': group_without_social_distancing.mean()
}

print(school_closure_data_summary, social_distancing_data_summary)

# Reusing the merged testing data from earlier analysis
# Calculate the correlation between testing rates and COVID-19 case trends

# Merging the testing data with the case data
merged_testing_case_data = pd.merge(merged_testing_data_clean, london_case_data, 
                                    on=['areaType', 'areaName', 'areaCode', 'date'], 
                                    how='inner')

# Calculating correlation between testing rates (number of new virus tests) and case trends (new cases by specimen date)
corr_new_tests_vs_new_cases, p_value_new_tests_vs_new_cases = pearsonr(
    merged_testing_case_data['newVirusTestsBySpecimenDate'],
    merged_testing_case_data['newCasesBySpecimenDateRollingSum']
)

# Calculating correlation between positivity rates and case trends
corr_positivity_rate_vs_new_cases, p_value_positivity_rate_vs_new_cases = pearsonr(
    merged_testing_case_data['uniqueCasePositivityBySpecimenDateRollingSum'],
    merged_testing_case_data['newCasesBySpecimenDateRollingSum']
)

# Displaying the correlation coefficients and p-values
corr_results_testing_case = {
    'Correlation (New Tests vs New Cases)': corr_new_tests_vs_new_cases,
    'P-Value (New Tests vs New Cases)': p_value_new_tests_vs_new_cases,
    'Correlation (Positivity Rate vs New Cases)': corr_positivity_rate_vs_new_cases,
    'P-Value (Positivity Rate vs New Cases)': p_value_positivity_rate_vs_new_cases
}

print(corr_results_testing_case)

# Load the hospital admissions and ventilation data files
hospital_admissions_file_path = '/Users/junrunchen/Downloads/data_2023-Dec-14 (6).csv'
ventilation_data_file_path = '/Users/junrunchen/Downloads/data_2023-Dec-14 (7).csv'

hospital_admissions_data = pd.read_csv(hospital_admissions_file_path)
ventilation_data = pd.read_csv(ventilation_data_file_path)

# Convert 'date' to datetime for time-series analysis
hospital_admissions_data['date'] = pd.to_datetime(hospital_admissions_data['date'])
ventilation_data['date'] = pd.to_datetime(ventilation_data['date'])

# Filter data for London
london_hospital_admissions = hospital_admissions_data[hospital_admissions_data['areaName'] == 'London']
london_ventilation_data = ventilation_data[ventilation_data['areaName'] == 'London']

# Time-series plot of hospital admissions in London
plt.figure(figsize=(15, 6))
plt.plot(london_hospital_admissions['date'], london_hospital_admissions['newAdmissions'], label='New Hospital Admissions')
plt.title('Hospital Admissions in London')
plt.xlabel('Date')
plt.ylabel('Number of Admissions')
plt.legend()
plt.grid(True)
plt.show()

# Time-series plot of mechanical ventilation bed occupancy in London
plt.figure(figsize=(15, 6))
plt.plot(london_ventilation_data['date'], london_ventilation_data['covidOccupiedMVBeds'], label='COVID-Occupied Mechanical Ventilation Beds')
plt.title('Mechanical Ventilation Bed Occupancy in London')
plt.xlabel('Date')
plt.ylabel('Number of Beds Occupied')
plt.legend()
plt.grid(True)
plt.show()

# Merging hospital admissions data with public health measures data
london_hospital_admissions['date'] = london_hospital_admissions['date'] - pd.to_timedelta(london_hospital_admissions['date'].dt.dayofweek, unit='d')
merged_hospital_measures_data = london_hospital_admissions.merge(public_health_measures_data, on='date', how='left')

# Performing hypothesis testing to assess the impact of public health measures on hospital admissions
# Example: Assessing the impact of school closures on hospital admissions
group_admissions_with_school_closure = merged_hospital_measures_data[merged_hospital_measures_data['schools_closed'] == 1]['newAdmissions']
group_admissions_without_school_closure = merged_hospital_measures_data[merged_hospital_measures_data['schools_closed'] == 0]['newAdmissions']

# Performing a t-test to compare the means of the two groups
t_stat_admissions_school_closure, p_value_admissions_school_closure = ttest_ind(group_admissions_with_school_closure, group_admissions_without_school_closure, equal_var=False, nan_policy='omit')

# Similarly, we can assess the impact of other public health measures like social distancing on hospital admissions
group_admissions_with_social_distancing = merged_hospital_measures_data[merged_hospital_measures_data['stay_at_home'] == 1]['newAdmissions']
group_admissions_without_social_distancing = merged_hospital_measures_data[merged_hospital_measures_data['stay_at_home'] == 0]['newAdmissions']

# Performing a t-test to compare the means of the two groups
t_stat_admissions_social_distancing, p_value_admissions_social_distancing = ttest_ind(group_admissions_with_social_distancing, group_admissions_without_social_distancing, equal_var=False, nan_policy='omit')

print(t_stat_admissions_school_closure, p_value_admissions_school_closure, t_stat_admissions_social_distancing, p_value_admissions_social_distancing)

# Load the death data file again for time-series analysis
death_data_file_path = '/Users/junrunchen/Downloads/data_2023-Dec-14 (14).csv'
death_data = pd.read_csv(death_data_file_path)

# Convert 'date' to datetime for time-series analysis
death_data['date'] = pd.to_datetime(death_data['date'])

# Filter data for London
london_death_data = death_data[death_data['areaName'] == 'London']

# Merging death data with public health measures data for correlation analysis
london_death_data['date'] = london_death_data['date'] - pd.to_timedelta(london_death_data['date'].dt.dayofweek, unit='d')
merged_death_measures_data = london_death_data.merge(public_health_measures_data, on='date', how='left')

# Calculating correlation between weekly deaths and public health measures such as school closures and social distancing
corr_deaths_vs_school_closure = merged_death_measures_data[['newWeeklyNsoDeathsByRegDate', 'schools_closed']].corr().iloc[0, 1]
corr_deaths_vs_social_distancing = merged_death_measures_data[['newWeeklyNsoDeathsByRegDate', 'stay_at_home']].corr().iloc[0, 1]

print(corr_deaths_vs_school_closure, corr_deaths_vs_social_distancing)



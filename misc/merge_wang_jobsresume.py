import pandas as pd

jobs_resumes_df = pd.read_csv("/nlp/scr/nmeist/EvalDims/modules/jobs_resumes.csv")
wang_df = pd.read_csv("/nlp/scr/nmeist/EvalDims/modules/wang.csv")

wang_df = wang_df[wang_df['model'] == 'GPT-4']


# Select 'Role' and 'Preprocessed_Resume' from wang_df, rename for target columns
wang_to_add = wang_df[['Role', 'Preprocessed_Resume']].copy()
wang_to_add = wang_to_add.rename(columns={'Role': 'job', 'Preprocessed_Resume': 'resume'})
wang_to_add['job description'] = ''
wang_to_add['source'] = 'wang'

# Only keep columns as in jobs_resumes_df (intersecting columns, preserving jobs_resumes_df order)
cols_to_use = [col for col in jobs_resumes_df.columns if col in wang_to_add.columns]
wang_to_add = wang_to_add[cols_to_use]

# Append the rows
jobs_resumes_df = pd.concat([jobs_resumes_df, wang_to_add], ignore_index=True)
jobs_resumes_df = jobs_resumes_df[['job', 'job description', 'resume', 'source']]

# read in karvonen
karvonen_df = pd.read_csv("/nlp/scr/nmeist/EvalDims/modules/karvonen.csv")
results = []

for i, row in karvonen_df.iterrows():
    resume_str = row['Resume_str']
    # the resume is the rest of the string after the job
    try: 
        job, resume = resume_str.split(' Summary  ')
        results.append({'job': job, 'job description': '', 'resume': resume, 'source': 'karvonen'})
    except: pass

jobs_resumes_df = pd.concat([jobs_resumes_df, pd.DataFrame(results)], ignore_index=True)
jobs_resumes_df = jobs_resumes_df[['job', 'job description', 'resume', 'source']]
jobs_resumes_df.to_csv("/nlp/scr/nmeist/EvalDims/modules/jobs_resumes.csv", index=False)
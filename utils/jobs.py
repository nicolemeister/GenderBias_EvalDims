import pandas as pd
import random
from collections import defaultdict


'''
Module if substituting in other jobs

import os

# RUN IF DOING JOBS V2

jobs_dir = "jobs_v2"
job_desc_dir = os.path.join(jobs_dir, "job_description")
resumes_dir = os.path.join(jobs_dir, "resumes")

jobs = []

for filename in os.listdir(job_desc_dir):
    if filename.endswith(".txt"):
        job_name = filename[:-4]
        # Read job description
        with open(os.path.join(job_desc_dir, filename), "r", encoding="utf-8") as f:
            job_description = f.read().strip()
        # Read resume
        resume_path = os.path.join(resumes_dir, filename)
        if os.path.exists(resume_path):
            with open(resume_path, "r", encoding="utf-8") as f:
                resume = f.read().strip()
        else:
            resume = ""
        jobs.append([job_name, job_description, resume])
        

'''

class Jobs:
    def __init__(self):
        # You can define different sets of names for different ids here if needed
        self.jobs_by_id = {
            'armstrong': [["cashier", "Cashiers process payments from customers purchasing goods and services.",
                        "CASHIER Results-oriented, strategic sales professional with two years in the Retail industry. Cashier who is highly energetic, outgoing and detail-oriented. Handles multiple responsibilities simultaneously while providing exceptional customer service. Reliable and friendly team member who quickly learns and masters new concepts and skills. Passionate about helping customers and creating a satisfying shopping experience. Core Qualifications: Cash handling accuracy, mathematical aptitude, organized, time management, detail-oriented, excellent multi-tasker, strong communication skills, flexible schedule, proficient in MS Office Skills: Calculators, Cash registers, Credit, debit, checks and money, Inventory, Sales, scanners, tables Experience: Cashier Sept 2021 - Current Receive payment by cash, check, credit cards, vouchers, or automatic debits. Issue receipts, refunds, credits, or change and process merchandise returns and exchanges. Greet customers, answer questions, resolve complaints and provide info on current policies. Maintain clean and orderly checkout areas and complete other general cleaning duties, such as mopping floors and emptying trash cans. Count money in cash drawers at the start of shifts to ensure correct amounts and adequate change. Calculate total payments received during a time period, and reconcile this with total sales. Inbound/Return Clerk June 2020 to Aug 2021 Changed equipment over to a new product. Maintained proper stock levels on a line. Helped achieve company goals by supporting production workers. Apparel Associate January 2019 to February 2020 Greet customers, help locate merchandise, and explain use and care of merchandise to customers. Compute sales prices, total purchases and receive and process cash or credit payment. Maintain knowledge of current promotions, policies, and security practices. Inventory stock, requisition new stock, and clean shelves, counters, and tables. Exchange merchandise for customers and accept returns. Open and close cash registers, performing tasks such as counting money, separating charge slips, coupons, and vouchers, balancing cash drawers, and making deposits. Education: Henry County High School: High School Diploma May 2020"
                        ], 
                        ["administrative assistant", "Secretaries and administrative assistants do routine clerical and organizational tasks. They arrange files, prepare documents, schedule appointments, and support other staff.",
                        "ADMINISTRATIVE ASSISTANT Enthusiastic student-teacher with superb leadership and communication skills. Easily cultivates trusting and productive relationships with students, parents, teachers administration, and others. Effective at fostering a positive working environment with excellent interpersonal and organizational skills. Skills: Communication, Interpersonal Skills, Management, Organization, Mastery of Microsoft Office Programs (Excel, Word, PowerPoint, Outlook), Attention to Detail, Public Speaking, Curriculum Development Experience: Administrative Assistant 06/2021 to 08/2021 Maintaining informational material and compiling information for reports. Setting up and maintaining supervised bank accounts monthly. Reconciling local office concentrated banking systems and preparing tax vouchers as applicable. Maintaining databases, conducting weekly computer backups, and monitoring secure file storage. Teacher Aide, Cooperative Education Trainee Program 09/2020 to 06/2021 Assisted in tutoring struggling students and worked with SPED students. Worked with teachers on curriculum to help improve both math and english departments. Made assignments to help students practice their skills and created educational games. Assisted in preparing materials for parent/teacher meetings and evaluating the progress of students throughout the school year.  Administration Office Assistant 06/2020 to 08/2020 Worked with the Director of the Cultural Affairs Department with filing papers, answering phone calls, assisting on historic preservation projects, and working with clients. Educated young students about the importance of preserving island culture and language. Student Activities Office Assistant 08/2018 to 3/2020 Helped organize activities in the University's Campus. Worked with other Universities to create combined events. Assisted clubs and organizations for sponsored activities volunteering opportunities Education: Bachelor of Arts: May 2022 Elementary Education University of Portland Portland, OR Kappa Delta Pi Honor Society Officer Portland Helping Hands and Family Homeless Shelter High School Diploma: May 2018 Rota High School Class Valedictorian, National Honor Society President, Youth Advisory President, Anti-Bullying Campaign President, Take Action Youth Advocacy Member, Junior Achievement Program Public Relations Officer, and Leadership Award Founded and led enrichment program at Elementary School: 'The Reading Bridge Project'"
                        ], 
                        ["chief executive officer", "Determine and formulate policies and provide overall direction of companies or private and public sector organizations within guidelines set up by a board of directors or similar governing body. Plan, direct, or coordinate operational activities at the highest level of management with the help of subordinate executives and staff managers.",
                        "CHIEF EXECUTIVE OFFICER Award-winning executive and marketing professional experienced in high-volume, multi-unit, retail and business operations in the pharmaceutical, financial services, and food and beverage industries. Demonstrated expertise in brand development, territory management, sales operations, product launches, recruiting, and business development. Skilled in utilizing technology as a tool to improve organizational efficiency. Desires a high-level position in a professional corporate environment. Skills: Brand Development, Project Management, Training & Development, Sales Operations, Merchandising Accomplishments: Increased annual sales to nearly $5.7 million through strategic marketing & sales campaigns. Launched aggressive growth plans that increased customer base from 0 to 15,000 customers. Created strategies to expand existing customer sales, resulting in 200% sales growth. Grew a targeted newsletter subscriber list from 0 to 6,000 members in just 12 months.Earned the Winner's Circle Award in 2008. Experience: Chief Executive Officer 10/2016 to Current Developed and launched Greenie Tots, a full-line of children's entrees, currently sold in mass retail including Whole Foods, Giant Eagle, Safeway, and independent grocery stores across US. Hired & trained brand ambassadors that marketed and sold brand to consumers & retail outlets. Managed production setup and distribution with large national natural products distributors. Developed incentive performance plan which motivated staff and resulted in 70% sales increase. Headed marketing and integrated advertising campaigns across multiple media platforms. Increased profits by 60% in one year through restructure of business line. Healthcare Management Representative 08/2011 to 09/2016 Responsible for a portfolio of billion dollar revenue medications to increase market base and change physician prescribing habits. Developed and maintained networks/partnerships with external partners such as physicians, hospitals, community advocacy groups, pharmacies, and corporate employers. Assisted District Manager with the development and leadership for district strategy for product launches, sales initiatives, and team motivational activities. Finance Intern 05/2010 to 08/2010 Maintained partnerships with external customers such as school institutions and businesses. Developed and maintained a customer database with client investment positions and future investment goals. Created visual tools to assist the VP in presenting to external partners. Education: MBA: Business Administration August 2011 Florida A&M University Tallahassee, FL BS: Business Administration August 2010 Florida A&M University Tallahassee, FL"
                        ],
                        ["financial analyst", "Financial analysts guide businesses and individuals in decisions about expending money to attain profit. They assess the performance of stocks, bonds, and other types of investments.",
                        "FINANCIAL ANALYST Highlights: Microsoft Excel and PowerPoint (intermediate), Capital IQ (intermediate), FactSet (intermediate), SNL (intermediate), Bloomberg (beginner/intermediate), SPSS (intermediate) Experience: Financial Analyst July 2015 to Current Support CEO and corporate operating committee directly by shaping and implementing AIG's strategy on a global level initiate Evaluate and execute M&A deals and innovation investments to enable AIG's inorganic growth Selected Transaction Experience and Strategic Projects: 500 Million Divestiture Work with senior management to identify ~$208 million of allocated versus direct expenses to make normalizing adjustments to pretax operating income (PTOI) and drive up valuation price of divested entity. Craft marketing language and organize flow of confidential information memorandum to prepare company leadership for management discussions with potential buyers. Write memos detailing industry dynamics and investment recommendations. Build database of financial metrics including market capitalization, total revenues and assets, and segment revenues from 70 companies to formulate a peer list and competitor set. Investment Banking Summer Analyst	 June 2014 to August 2014 Supported Industrials Coverage Group by building client presentations, evaluating and compiling financial metrics and aiding in model analysis and valuation. Compared management and board structure in 9 peer filings in order to help senior leadership in the drafting of the prospectus and road show materials 170 million Buy-Side M&A. Compiled 8 years of titanium price and production data to project summary financials and aid private equity buyer in determining valuation price of the target company. Built 50+ acquisition target profiles to help group pitch M&A opportunities to 5 different clients. Finance Intern September 2013 to December 2014 Worked in Analytics and Sales gaining exposure to equities, fixed income, and commodities. Operated Bloomberg Terminal to obtain data for 10 stock pitches and technical analysis of securities and industry overviews. Education: Bachelor of Science: Economics and Psychology May 2015 Yale University New Haven, CT GPA: 3.75/4.0 Relevant Coursework: Strategic Management, Accounting & Valuation Cumulative SAT score: 2390 Interests: Traveling, piano, violin, table tennis, swimming, volunteering, pistachio ice cream Skills: Accounting, budgeting, commodities, clients, database, equities, financial statements, innovation, investments, leadership, marketing, math, strategic management, strategic planning, valuation"
                        ], 
                        ["human resources specialist", "Human resources specialists recruit, screen, and interview job applicants and place newly hired workers in jobs. They also may handle compensation and benefits, training, and employee relations.",
                        "HUMAN RESOURCES SPECIALIST Quality-driven analytical professional who delivers consistent and successful results in HR affairs, including recruitment and retention, staff development, safety and health, conflict resolution, benefits and compensation, HR audit and records management, HR policies development, and legal compliance. Experience: HR Generalist 09/2019 to Current Coordinate HR Support to five Resorts with 2000+ employees. Manage employment, wage and salary administration, benefits, training, employee/labor relations, organizational development, and payroll. Work closely with Resorts General Manager to achieve company goals and objectives. Created and implemented a training program for managers and supervisors, including employee motivation, effective leadership, completing disciplinary actions, and performance reviews. Satisfied record keeping requirements evaluated during annual HR Audit. Reduce turnover rate by improving recruitment with effective interviewing and proper selection. Office Manager 08/2017 to 08/2019 Responsible for recruiting, interviewing, hiring, and monitoring payroll for 60+ retail employees. Maintain HRIS database, conduct reference checks, and perform new hire and safety orientation. Ran and audited weekly benefits reports. Issue monthly and quarterly workers compensation reports to senior management. Maintained OSHA logs and acted as a liaison between the carrier and the injured employees. Acted as a liaison between benefit vendors and employees to resolve and troubleshoot issues. Provide assistance to the Benefits Manager in creating a company-wide wellness program. HR Specialist 05/2016 to 08/2017 Worked with senior management to create HR policies and procedures; recruit employees; create group benefits databases; and develop orientation, training, and incentive programs. Personal efforts were cited as the driving force behind the branch's employee-retention rate of 89% within an industry where high turnover is the norm. Devise creative, cost-effective incentives and morale-boosting programs (including special events and a tiered awards structure) that increased employee satisfaction and productivity. Reduced benefits costs by 15% annually through meticulous record keeping and ensuring that the company did not pay for benefits for which employees were ineligible. Education: Master of Science: Industrial/Organizational Psychology 2019 University of Maryland Baltimore, MD BA: Business - Human Resources 2016 Arizona State University Tempe, AZ Skills: Benefits, budget, databases, employee relations, special events, hiring, HRIS, insurance, labor relations, leadership, managing, Access, Excel, MS Office, Outlook, PowerPoint, Word, Oracle, organizational development, payroll, performance reviews, personnel, record keeping, recruiting, reporting, teamwork"
                        ],
                        ["software developer", "Software developers create computer applications that allow users to do specific tasks and the underlying systems that run the devices or control networks. Software quality assurance analysts and testers design and execute software tests to identify problems and learn how the software works.",
                        "SOFTWARE DEVELOPER Experienced software engineer with 7+ years of product development experience in broadcast media, editing software, and engineering technologies. Experience: Software Engineer 09/2018 to Current Used Visual C++, Windows, STL, OOP, MFC, threads, file maps, ATL, IPC, FTP, TCP, HTTP, XML, JSON, services, web services, REST API, SOA, media formats and standards, codec SDK and integration, MPEG-DASH, API design and documentation. Mentored core editing team to size of 8 and collaborated with expanded, international team. Received 'Reuse Innovation Award' for IP reuse, major factor to Server business unit turnaround. Released deliverables for Nexio Software Suite 6.0, 7.0, 8.0 and Global Proxy Suite 2.5, 3.0, 3.5 Team helped with providing C# web service framework, CLI bridge layer, and device testing. Published functional, REST API, and URI specification document. Developed Content Manager, Helper, Picon, and Requestor, improved Encoder, GPRX, Helios, MB, Scavenger, and Transcoder, and guided GPRX, Helper, and Requestor to completion. Achieved constant UI performance under a few milliseconds regardless of user operation. Proposed to introduce architecture in ftp-server Approach helped to consolidate all media formats as one product build, and scalability.Software Developer 06/2015 to 8/2018 Designed and implemented fundamental DLL components for evolution of video editing products, and major projects include (C++, Win32, threads, GUI) Created interactive playback architecture Media file reader and writer components and scalable decoder and encoder architectures. Worked on Video and audio rendering engines Hardware integration modules critical part to business success in post-production space. Helped with Interactive picon and waveform drawing components 64-bit and Unicode migration of all modules with over 4 million lines of code. Software Engineer Intern 05/2014 to 08/2014 Helped create video editing software Velocity for post-production space. Developed projects including media management tools, Render Bank, and video effects. Education BS: Computer Science 2015 Illinois Institute of Technology Illinois, IL Skills: API, Approach, ATL, audio, backup, broadcast, C++, CLI, Hardware, Content, clients, documentation, dynamic HTML5, editing, XML, FTP, functional, GUI, http, IDs, Innovation, explorer, IP, JavaScript, json, LAN, MB, C#, MFC, Windows, NAS, OOP, Proxy, rendering, SAN, Scrum, servers, specification, team management, threads, troubleshooting, Video, Video Editing, Visual C++, workflow"
                        ],
                        ["social worker", "Social workers help individuals, groups, and families prevent and cope with problems in their everyday lives. Clinical social workers diagnose and treat mental, behavioral, and emotional problems.",
                        "SOCIAL WORKER Leadership: YMCA Camp Orkila (Orcas Island, WA) Adventure Team Facilitator, 2017 Completed a 5-day intensive training on challenge course facilitation; pushed youth to identify their limits and challenge themselves, and led reflection activities to encourage student learning. Chosen by Girls LEAD to facilitate a group of campers focused on leadership, service, and wellness; aided in program coordination; encouraged youth self-awareness and self-confidence. Experience: School Social Worker 	Sep 2021 to Current Provide one-on-one and group counseling to students. Facilitate extended day program for 15 high school students, providing support and assisting in the development of school success skills in a safe, positive learning environment. Serve as liaison between school and family to increase access to info. Partner with teachers to support student academic and socio-emotional growth. Maintain accurate and up-to-date records; organize meetings with teachers, counselors, and administrators to meet student need; and provide referrals for needed services. Achieve positive outcomes in participant recruitment, retention, and overall academic performance; develop strong relationships with students, families, and school staff. Child Family Advocate Aug 2019 to Aug 2021 Worked with community leaders and public agencies to promote community service programs. Directed protective placement, case management, and family reunification activities. Created and implemented developmentally-appropriate curriculum to address all learning styles. Advised patients on community resources, made referrals, and devised realistic treatment plans. Communicated with public social and welfare agencies to obtain and provide information. Civic Engagement Intern Jun to Aug 2018 Organized and managed volunteer engagement and voter registration. Coordinated meetings with allied community organizations; facilitated voter registration at naturalization ceremonies; canvassed for the Driver Card campaign; and registered new voters. Managed and maintained volunteer database and volunteer recruitment efforts. Education: Masters: Social Work May 2021 University of Washington Bachelors of Arts: Environmental Humanities Politics 2019 Whitman College Magna Cum Laude GPA: 3.815 Jan Meier Award for Best Essay in Environmental Studies, 2019 Lomen-Douglas Scholarship, 2018 Myers-Little Scholarship, 2018 Skills: Administrative, counseling, database, leadership development, mentoring, problem solving, programming"
                        ],
                        ["secondary teacher", "High school teachers help prepare students for life after graduation. They teach academic lessons and various skills that students will need to attend college or to enter the job market.",
                        "Certified to teach all general sciences. Experienced in teaching biology, chemistry, and physical science, excellent organizational skills, and proficient with instruction, testing, grade, and attendance software. Experience: Physics Teacher (09/2021 to Current) Execute, implement, and modify lesson plans while incorporating differentiated instruction. Design and align lessons, labs, and assessments incorporating STEM, problem based learning, Common Core, and NGSS. Volunteer and participate in schools extracurricular activities such as selling tickets for the school talent show and participating in the 5K for the scholarship fund. Co-teach with special education teachers while incorporating student IEP and 504 plans. Plan and present Google applications training for high school professional development. Biology Teacher (09/2019 to 6/2021) Implement 9th grade biology lesson plans and bring ideas, practices, and theories from professional development workshops into the classroom. Communicate with parents/guardians regarding student progress within the classroom. Regularly participate in professional development opportunities including NJEA and NSTA Conventions and training programs in equipment maintenance and Good Laboratory Practices. Successfully fundraised money for incorporating TI Nspire CX graphing calculators into the classroom through DonorsChoose.org. Achieved Level 1 Google Certified Educator status in October 2021. Student Teacher (01/2019 to 03/2019) Develop labs, assignments, and projects to reinforce material taught previously, encourage deeper understanding, and bridge multiple disciplines, such as writing, science, and social sciences. Strive for educational improvement by applying constructive criticism to lessons during student teaching experience. Help with formative and summative assessments for content related to the state standards. Education: Teaching Certification Program: Secondary Education May 2019 University of Washington Seattle, WA Collaborate with teachers to incorporating STEM field research into classroom lessons Participate in research on barriers to student belonging in STEM education BS: Biological Sciences May 2018 Drexel University Philadelphia, PA"
                        ],
                        ["nursing assistant", "Nursing assistants, sometimes called nursing aides, provide basic care and help patients with activities of daily living. Orderlies transport patients and clean treatment areas.",
                        "NURSING ASSISTANT Certified Nursing Assistant with 4+ years work experience in a fast-paced environment handling confidential paperwork, administering medication and providing quality, empathetic, patient-focused care, monitored vital signs, assisted with feeding, bathing/grooming, positioning and range of motion exercises. Highly compassionate and looking for a long term care position. Professional Experience: Nursing Assistant 08/2020 to Current Tended to patients with chronic illnesses and provided them with emotional support. Recognized and reported abnormalities in patients' health status to nursing staff. Provided personal nursing care in pre- and post-operative situations. Assisted with transferring patients in and out of wheelchairs and adaptive equipment. Read and recorded temperature, pulse, respiration and blood pressure. Nursing Assistant 01/2018 to 07/2020 Took vitals and charted daily information on the residents such as mood changes, mobility activity, eating percentages, and daily inputs and outputs. Provided patients and families with emotional support. Prepared patient rooms prior to their arrival and cleaned patients' living quarters. Positioned residents for comfort and to prevent skin pressure problems. Healthcare Assistant 06/2018 to 12/2018 Observed and documented patient status and reported patient complaints to case manager. Completed and submitted clinical documentation in accordance with agency guidelines. Assisted with adequate nutrition and fluid intake. Provided companionship and emotional support to patients and families. Performed household tasks such as laundry, dusting, washing dishes and vacuuming. Education and Training: Licenses: CPR Certification, First Aid Certification, Environmental Emergencies Certification, Adult/Child CPR With Mask Certification through the American Heart Association Professional Healthcare In-Service 2018: Adult Behavioral and Diagnosed Mental Health Disorders Alzheimer's Association High School Diploma: General Lebanon High School Skills: Patient-focused care, Excellent interpersonal skills, Compassionate and trustworthy caregiver, Time management, Effectively interacts with patients and families, Medical terminology, Hospice care provider, Wound care, Charting and record keeping Interests: Running, Reading, Painting, Playing the Piano, Yoga"
                        ],
                        ["mechanical engineer", "Mechanical engineers research, design, develop, build, and test mechanical and thermal sensors and devices, including tools, engines, and machines.",
                        "MECHANICAL ENGINEER CAD, CAM, Finite Element Analysis, Mechanical Design, Product Design and Development Skills Skills: 5 years of experience with CAD packages (SolidWorks, Autodesk Inventor, AutoCAD, CATIA) 3 years of experience with CAE Softwares (HyperMesh, Abaqus, ANSYS, Optistruct) 3 years of experience with Analysis (Linear & Non-linear Static, Dynamic, Design Optimization) Experience with design for manufacturing, generating Bill of Materials, DFMEA, Sculpting Experience with advanced material selection for prototyping, manufacturing, and 3D printing Experience: Mechanical Engineer 9/2019 to Current Finite Element Analysis of Industrial Robotic Assembly Designed a 6-axis SCARA Robot for pick and place operation in automotive industry. Performed static analysis with stainless steel 304 to evaluate the maximum load. Optimized design using OptiStruct by varying mesh sizes and element order. Simulated assembly with dynamic analysis to find distorted elements and verify structure. Reliability Engineering Analysis on Automotive Oil Pump Used industrial reliability specifications to select power consumption and flow rate at three distinct levels of rpm to study its variability Conducted Failure Mode Effect Analysis (FMEA) to analyze potential causes of failures Provided vegetable oil as a coolant, which resulted in unburnt and recyclable chips. Mechanical Engineering Intern 06/2018 to 08/2018 Initiated a project to perform a failure investigation in mufflers due to the low clearance of roads. Established and coordinated maintenance, safety procedures, and supply of materials. Developed failure reports including feedback based on common failures in automotive industry. Set up and calibrated accelerometers on Hyundai cars to conduct tests to analyze the modes of vibration of the vehicle and the steering column. Manufacturing Engineer Intern	 05/2017 to 07/2017 Analyzed automation, process parameters, different equipment to shape and control the profile of chips, and manufacturing process of Hot Strip Coil. Re-designed the existing shop floor to improve space utilization, increase material flow, optimize labor, reduce holding costs by 5%, and improve space utilization by 20%. Performed statistical analysis on historical data of the operating parameters to identify significant factors contributing to process deviation and affecting the cold crushing strength of the pellet. Generated Bill of Materials and calculated overall manufacturing cost. Education: BS: Mechanical & Aerospace Engineering  May 2019 Illinois Institute of Technology Chicago, IL GPA: 3.5/4.0 Certifications: Autodesk Inventor Professional Software and AutoCAD Software"
                        ]],
            'rozado': None, 
            'gaeb': None
            }
        

    def get_jobs_resumes(self, id, exp_framework, random_state=42):
        """
        Given an input id, return a dictionary mapping race to a list of names.
        You can customize the mapping per id if needed.
        """
        # read in 

        jobs_resumes = pd.read_csv('modules/jobs_resumes.csv')
        import random
        random.seed(random_state)


        jobs_resumes = jobs_resumes[jobs_resumes['source'] == id]
        if exp_framework == 'armstrong':

            # Get the unique jobs
            all_unique_jobs = jobs_resumes['job'].unique().tolist()

            # Randomly sample 10 unique jobs (make sure there are at least 10 jobs) # TO DO: NEED TO ADJUST THIS BASED ON UPSAMPLING/DOWNSAMPLING FLAG 
            sample_size = min(10, len(all_unique_jobs))
            sampled_jobs = random.sample(all_unique_jobs, sample_size)

            # Find the indices of the sampled jobs in the dataframe
            jobs = []
            job_descriptions = []
            resumes = []

            for job in sampled_jobs:
                # Randomly sample one occurrence of this job in the dataframe
                row = jobs_resumes[jobs_resumes['job'] == job].sample(n=1, random_state=random_state).iloc[0]
                jobs.append(row['job'])
                job_descriptions.append(row['job description'])
                resumes.append(row['resume'])
            return jobs, job_descriptions, resumes

        elif exp_framework == 'rozado':
            # goal: return a pair of resumes for each job description possible (rozado and wang)

            # for each job, report two of the same resumes (only difference will be names) 
            if id == 'armstrong':
                jobs = jobs_resumes['job'].tolist()
                job_descriptions = jobs_resumes['job description'].tolist()
                resumes = jobs_resumes['resume'].tolist()
                return jobs, job_descriptions, resumes

            elif id == 'rozado' or id == 'wen' or id == 'wang':
                pass
        elif exp_framework=='yin':
            # Get the unique jobs
            all_unique_jobs = jobs_resumes['job'].unique().tolist()

            # Randomly sample 4 unique jobs - pick these jobs based on ones that appear the mostly frequently in the set
            job_counts = jobs_resumes['job'].value_counts()
            most_frequent_jobs = job_counts.index.tolist()

            # grab the 4 most frequent jobs. if all counts in most_frequent_jobs are the same, then sample the rest of the jobs until we have 4 unique jobs. 
            if len(set(job_counts)) == 1 and len(all_unique_jobs) >= 4:
                sampled_jobs = random.sample(all_unique_jobs, 4)
            else:
                sampled_jobs = most_frequent_jobs[:4]

            # Find the indices of the sampled jobs in the dataframe
            jobs = []
            job_descriptions = defaultdict(list)
            resumes = defaultdict(list)

            for job in sampled_jobs:
                jobs.append(job)
                # Get all occurrences of this specific job
                job_rows = jobs_resumes[jobs_resumes['job'] == job]
                
                # Determine if we need to sample with replacement
                # (If available rows < 8, we must replace. If >= 8, we prefer not to replace)
                target_n = 8
                replace_strategy = len(job_rows) < target_n
                
                # Sample the rows
                sampled_rows = job_rows.sample(n=target_n, replace=replace_strategy, random_state=random_state)
                
                for resume in sampled_rows['resume'].tolist():
                    resumes[job].append(resume)

                for job_description in sampled_rows['job description'].tolist():
                    job_descriptions[job].append(job_description)


            return jobs, job_descriptions, resumes
        
        else:
            raise ValueError(f"experimental_framework {exp_framework} not supported")




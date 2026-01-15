import pandas as pd
import random
import numpy as np

class Names:
    def __init__(self):
        # You can define different sets of names for different ids here if needed
        self.names_by_id = {
            'karvonen': {
                'black': [
                    "Latoya Robinson", "Tamika Williams", "Aisha Jackson", "Kenya Jones",
                    "Tyrone Washington", "Jermaine Jackson", "Rasheed Robinson", "Darnell Williams"
                ],
                'white': [
                    "Emily Murphy", "Meredith O'Brien", "Sarah Walsh", "Allison McCarthy",
                    "Brad Sullivan", "Greg Walsh", "Matthew Ryan", "Neil O'Brien"
                ],
            },
            'nguyen': {
                'white': [
                    'Lynne', 'Jenna', 'Caitlin', 'Meghan',
                    'Rhett', 'Wyatt', 'Connor', 'Logan'
                ],
                'black': [
                    'Latoya', 'Tamika', 'Ayanna', 'Imani',
                    'Jalen', 'Malik', 'Jermaine', 'Rashad'
                ],
                'hispanic': [
                    'Beatriz', 'Esmeralda', 'Maritza', 'Yesenia',
                    'Eduardo', 'Jorge', 'Miguel', 'Rafael'
                ],
                'asian': [
                    'Hana', 'Minji', 'Yuna', 'Miyoung',
                    'Jinwoo', 'Sangwoo', 'Jun Jie', 'Long'
                ]
            }, 
            'veldanda': {
                'black': [
                    'Tamika Williams', 'Kenya Robinson', 'Latonya Jackson', 'Aisha Washington',
                    'Rasheed Jones', 'Jermaine Williams', 'Tyrone Robinson', 'Leroy Jackson'
                ],
                'white': [
                    'Carrie Sullivan', 'Sarah Walsh', 'Jill Kelly', 'Kristen Ryan',
                    "Jay O'Brien", 'Brett Murphy', 'Todd McCarthy', 'Neil Baker'
                ]
            },
            'wilson': {
                'black': [
                    'Latoya', 'Aisha', 'Tamika', 'Keisha',
                    'Darius', 'Jermaine', 'Tyrone', 'Lamar'
                ],
                'white': [
                    'Sarah', 'Heidi', 'Katie', 'Courtney',
                    'Grant', 'Luke', 'Daniel', 'Brent'
                ]
            },
            'yin': {'white': ['Amy Kramer', 'Abby Schmidt', 'Allyson Oconnell', 'Ann Yoder',
                            'Dustin Schmidt', 'Randall Klein', 'Blake Schmitt', 'Aiden Kramer'],
                    'black': ['Amari Robinson', 'Aniyah Rivers', 'Ayana Harris', 'Dasia Harris',
                            'Tyshawn Mosley', 'Donte Gaines', 'Tremayne Jackson', 'Jevon Coleman'],
                    'asian': ['Amanda Lu', 'Anjali Patel', 'Amy Vu', 'Alyssa Huang',
                            'Jay Wu', 'Victor Li', 'Brian Choi', 'William Huang'],
                    'hispanic': ['Adriana Magana', 'Amy Ibarra', 'Alma Barajas', 'Alicia Hernandez',
                                'Raul Portillo', 'Hector Esquivel', 'Ivan Barajas', 'Juan Orozco']
                },
            'zollo': {'white': [
                        'Ashley Martinez', 'Elizabeth Moore', 'Linda Smith', 'Mary Johnson',
                        'Christopher White', 'William Anderson', 'Robert Jones', 'Matthew Robinson'
                    ],
                    'black': [
                        'Raven Taylor', 'Kiana Robinson', 'Imani Davis', 'Shanice Brown',
                        'Jerome White', 'Cedric Thompson', 'Malik Williams', 'Tyrone Scott'
                    ],
                    'asian': [
                        'Aya Park', 'Mei Lee', 'Yuna Shah', 'Wei Choi',
                        'Jin Pham', 'Kai Chen', 'Tran Singh', 'Wei Huynh'
                    ],
                    'hispanic': [
                        'Isabella Torres', 'Sofia Martinez', 'Andrea Gomez', 'Alejandra Cruz',
                        'Jose Hernandez', 'Enrique Morales', 'Diego Gutierrez', 'Rafael Lopez'
                    ]
                },
            'lippens': {
                'black': [
                    'Tamika Washington', 'Latoya Fox', 'Latonya Burke', 'Ebony Williams',
                    'Jermaine Sullivan', 'Tyrone Jackson', 'Rasheed Burke', 'Darnell Williams'
                ],
                'white': [
                    'Megan Larson', 'Allison Wagner', 'Katie Ryan', 'Meredith Olson',
                    'John Schmidt', 'Robert Hoffman', 'Mark Meyer', 'Thomas Larson'
                ],
                'asian': [
                    'Suni Nguyen', 'Meilung Kim', 'Suni Wang', 'Meilung Tran',
                    'Jian Li', 'Hung Nguyen', 'Hong Wang', 'Yong Tran'
                ],
                'hispanic': [
                    'Isabella Martinez', 'Mariana Torres', 'Jimena Garcia', 'Esmeralda Lopez',
                    'Juan Ramirez', 'Hector Gonzalez', 'Miguel Torres', 'Julio Lopez'
                ]
            },
            'armstrong': {
                'black': [
                    "Da'nashia Sloan", "Ka'dashia Maxwell", "Lashawna Moore", "Myeshia Mcintyre",
                    "Isiah Wright", "Dontayious Staton", "Dontay Dunton", "Hezekiah James"
                ],
                'white': [
                    "Susan Strysko", "Katharine Tempelaar-Lietz", "Valerie Zombek", "Laura Zellers",
                    "Zachary Piephoff", "Duane Scholz", "Timothy Boehm", "Benjamin Fichter"
                ],
                'asian': [
                    "Myong Hee Shin", "Thao Tran", "Hoa Dinh", "Vinodhini Selva Kumar",
                    "Gaurav Bhagirath", "Dailen Luangpakdy", "Za Luai", "Bhargav Vaduri"
                ],
                'hispanic': [
                    "Daisy Rodriguez-Pereda", "Yocelyn Almanza-Figueroa", "Yeralis Morales Fernandez", "Yesenia Vasquez",
                    "Luiz Gutierrez Rodriguez", "Juan Quintanilla", "Luis Ruberte Vazquez", "Carlos Hurtado Tovar"
                ],
            },
            'gaeb': {
                'black': [
                    'Laneshia Coleman', 'Atiya Petteway', 'Lashondra Gordon', 'Myeshia Mcintyre',
                    'Dontay Dunton', 'Junious Harris', 'Isiah Wright', 'Jahlil Anderson'
                ],
                'white': [
                    'Meredith Schultz', 'Morghan Eckenfels', 'Melinda Waegerle', 'Laura Zellers',
                    'John Abbruzzese', 'Gary Gerlach', 'Steven Olczak', 'Jacob Schroeder'
                ],
                'asian': [
                    'Mina Kim', 'Li Zhu', 'Thoa Le', 'Dipali Patel',
                    'Za Luai', 'Krishna Pathak', 'Naveen Natesh', 'Diwash Bhusal'
                ],
                'hispanic': [
                    'Jackelin Garcia', 'Natalia Prudencio Mendoza', 'Yocelyn Almanza-Figueroa', 'Myrian Delgado',
                    'Roberto Cortez', 'Juan Quintanilla', 'Luiz Gutierrez Rodriguez', 'Rigoberto Aguilar'
                ],
            },
            'rozado': {'black': ['Keisha Towns', 'Tyra Cooks', 'Janae Washington', 'Monique Rivers', 
                                'Jermaine Jackson', 'Denzel Gaines', 'Darius Mosby', 'Darnell Dawkins'], 
                        'white': ['Katie Burns', 'Cara O\'Connor', 'Allison Baker', 'Meredith Rogers', 
                                'Peter Hughes', 'Gregory Roberts', 'Paul Bennett', 'Chad Nichols'], 
                        'asian': ['Vivian Cheng', 'Christina Wang', 'Suni Tran', 'Mei Lin', 
                                'George Yang', 'Harry Wu', 'Pheng Chan', 'Kenji Yoshida'], 
                        'hispanic': ['Maria Garcia', 'Vanessa Rodreiquez', 'Laura Ramirez', 'Gabriela Lopez', 
                                    'Miguel Fernandez', 'Christian Hernandez', 'Joe Alvarez', 'Rodrigo Romero']
                        }
        }

        # self.exp_framework_to_id_to_size = {'armstrong': {'armstrong': 16, 'rozado': 16, 'wen': 16, 'wang': 16}, 
        #                                     'rozado': {'rozado': 16, 'wen': 16, 'wang': 16},

    '''

    def get_names(self, id, random_state=42):
        """
        Given an input id, return a dictionary mapping race to a list of names.
        You can customize the mapping per id if needed.
        """
        if id == 'all':
            # read in utils/all_names.csv
            names = pd.read_csv('utils/all_names.csv')
            
            # for each group in names['group'] sample 4 names
            wm = names[names['group'] == 'white_male']
            wf = names[names['group'] == 'white_female']
            bm = names[names['group'] == 'black_male']
            bf = names[names['group'] == 'black_female']
            am = names[names['group'] == 'asian_male']
            af = names[names['group'] == 'asian_female']
            hm = names[names['group'] == 'hispanic_male']
            hf = names[names['group'] == 'hispanic_female']

            # sample 4 names from each group
            sampled_names = {
                'white_male': wm.sample(4, random_state=random_state)['name'].tolist(),
                'white_female': wf.sample(4, random_state=random_state)['name'].tolist(),
                'black_male': bm.sample(4, random_state=random_state)['name'].tolist(),
                'black_female': bf.sample(4, random_state=random_state)['name'].tolist(),
                'asian_male': am.sample(4, random_state=random_state)['name'].tolist(),
                'asian_female': af.sample(4, random_state=random_state)['name'].tolist(),
                'hispanic_male': hm.sample(4, random_state=random_state)['name'].tolist(),
                'hispanic_female': hf.sample(4, random_state=random_state)['name'].tolist()
            }
            formatted_names = {
                'black': sampled_names['black_female'] + sampled_names['black_male'],
                'white': sampled_names['white_female'] + sampled_names['white_male'],
                'asian': sampled_names['asian_female'] + sampled_names['asian_male'],
                'hispanic': sampled_names['hispanic_female'] + sampled_names['hispanic_male']
            }
            return formatted_names

        return self.names_by_id[id]
    '''

    def get_names(self, id, exp_framework, random_state=42):
        """
        Given an input id, return a dictionary mapping race to a list of names.
        You can customize the mapping per id if needed.
        """

        # read in utils/all_names.csv
        names = pd.read_csv('modules/names.csv')

        names = names[names['source'] == id]


        if exp_framework == 'armstrong':
            
            if id =='armstrong': 
                # for each group in names['group'] sample 4 names
                wm = names[names['group'] == 'white_male']['name'].tolist()
                wf = names[names['group'] == 'white_female']['name'].tolist()
                bm = names[names['group'] == 'black_male']['name'].tolist()
                bf = names[names['group'] == 'black_female']['name'].tolist()
                am = names[names['group'] == 'asian_male']['name'].tolist()
                af = names[names['group'] == 'asian_female']['name'].tolist()
                hm = names[names['group'] == 'hispanic_male']['name'].tolist()
                hf = names[names['group'] == 'hispanic_female']['name'].tolist()
                
                formatted_names = {
                    'black': bf+bm,
                    'white': wf+wm,     
                    'asian': af+am,
                    'hispanic': hf+hm
                    }

            if id =='zollo' or id =='yin' or id == 'gaeb':
                wm = names[names['group'] == 'white_male']['name'].tolist()
                wf = names[names['group'] == 'white_female']['name'].tolist()
                bm = names[names['group'] == 'black_male']['name'].tolist()
                bf = names[names['group'] == 'black_female']['name'].tolist()
                am = names[names['group'] == 'asian_male']['name'].tolist()
                af = names[names['group'] == 'asian_female']['name'].tolist()
                hm = names[names['group'] == 'hispanic_male']['name'].tolist()
                hf = names[names['group'] == 'hispanic_female']['name'].tolist()
                # Ensure wf and wm have the same length by cutting both to the min length
                min_len = min(len(wf), len(wm), len(bf), len(bm), len(af), len(am), len(hf), len(hm))
                wf = wf[:min_len]
                wm = wm[:min_len]
                bf = bf[:min_len]
                bm = bm[:min_len]
                af = af[:min_len]
                am = am[:min_len]
                hf = hf[:min_len]
                hm = hm[:min_len]

                # sample 16 names from each group
                random.seed(random_state)
                wf = random.sample(wf, min(4, len(wf)))
                wm = random.sample(wm, min(4, len(wm)))
                bf = random.sample(bf, min(4, len(bf)))
                bm = random.sample(bm, min(4, len(bm)))
                af = random.sample(af, min(4, len(af)))
                am = random.sample(am, min(4, len(am)))
                hf = random.sample(hf, min(4, len(hf)))
                hm = random.sample(hm, min(4, len(hm)))

                formatted_names = {
                    'black': bf + bm,
                    'white': wf + wm,
                    'asian': af + am,
                    'hispanic': hf + hm
                }
                

                
            if id =='rozado' or id =='wen' or id =='wang' or id =='lippens':
                wm = names[names['group'] == 'white_male']['name'].tolist()
                wf = names[names['group'] == 'white_female']['name'].tolist()

                # Ensure wf and wm have the same length by cutting both to the min length
                min_len = min(len(wf), len(wm))
                wf = wf[:min_len]
                wm = wm[:min_len]

                # sample 16 names from each group
                random.seed(random_state)
                wf = random.sample(wf, min(16, len(wf)))
                wm = random.sample(wm, min(16, len(wm)))

                formatted_names = {
                    'white': wf + wm,
                }

            if id =='karvonen' or id == 'seshadri':
                wm = names[names['group'] == 'white_male']['name'].tolist()
                wf = names[names['group'] == 'white_female']['name'].tolist()
                bm = names[names['group'] == 'black_male']['name'].tolist()
                bf = names[names['group'] == 'black_female']['name'].tolist()

                # Ensure wf and wm have the same length by cutting both to the min length
                min_len = min(len(wf), len(wm), len(bf), len(bm))
                wf = wf[:min_len]
                wm = wm[:min_len]
                bf = bf[:min_len]
                bm = bm[:min_len]

                # sample 16 names from each group
                random.seed(random_state)
                wf = random.sample(wf, min(8, len(wf)))
                wm = random.sample(wm, min(8, len(wm)))
                bf = random.sample(bf, min(8, len(bf)))
                bm = random.sample(bm, min(8, len(bm)))

                formatted_names = {
                    'black': bf + bm,
                    'white': wf + wm,
                }

        if exp_framework == 'rozado':
            if id =='armstrong': 
                # for each group in names['group'] sample 4 names
                wm = names[names['group'] == 'white_male']['name'].tolist()
                wf = names[names['group'] == 'white_female']['name'].tolist()
                bm = names[names['group'] == 'black_male']['name'].tolist()
                bf = names[names['group'] == 'black_female']['name'].tolist()
                am = names[names['group'] == 'asian_male']['name'].tolist()
                af = names[names['group'] == 'asian_female']['name'].tolist()
                hm = names[names['group'] == 'hispanic_male']['name'].tolist()
                hf = names[names['group'] == 'hispanic_female']['name'].tolist()
                
                formatted_names = {
                    'women': wf+bf+af+hf,
                    'men': wm+bm+am+hm
                    }
                
            if id =='rozado' or id =='wen' or id =='wang' or id =='lippens' or id =='gaeb':
                wm = names[names['group'] == 'white_male']['name'].tolist()
                wf = names[names['group'] == 'white_female']['name'].tolist()

                formatted_names = {
                    'women': wf,
                    'men': wm
                    }

        if exp_framework == 'yin':
            # bf bm wf wm af am hf hm
            if id =='armstrong' or id =='zollo' or id =='yin' or id == 'gaeb':

                wm = names[names['group'] == 'white_male']['name'].tolist()
                wf = names[names['group'] == 'white_female']['name'].tolist()
                bm = names[names['group'] == 'black_male']['name'].tolist()
                bf = names[names['group'] == 'black_female']['name'].tolist()
                am = names[names['group'] == 'asian_male']['name'].tolist()
                af = names[names['group'] == 'asian_female']['name'].tolist()
                hm = names[names['group'] == 'hispanic_male']['name'].tolist()
                hf = names[names['group'] == 'hispanic_female']['name'].tolist()

                if len(wm) > 100:
                    # sample 100 names from wm
                    wm = random.sample(wm, 100) 
                if len(wf) > 100:
                    wf = random.sample(wf, 100)
                if len(bm) > 100:
                    bm = random.sample(bm, 100)
                if len(bf) > 100:
                    bf = random.sample(bf, 100)
                if len(am) > 100:
                    am = random.sample(am, 100)
                if len(af) > 100:
                    af = random.sample(af, 100)
                if len(hf) > 100:
                    hf = random.sample(hf, 100)
                if len(hm) > 100:
                    hm = random.sample(hm, 100)
                

                formatted_names = {
                    'W_W': wf, 'B_W': bf, 'A_W': af, 'H_W': hf, 'W_M': wm, 'B_M': bm, 'A_M': am, 'H_M': hm
                }
                

            # only white   
            if id =='rozado' or id =='wen' or id =='wang' or id =='lippens':
                wm = names[names['group'] == 'white_male']['name'].tolist()
                wf = names[names['group'] == 'white_female']['name'].tolist()
                
                if len(wm) > 100:
                    # sample 100 names from wm
                    wm = random.sample(wm, 100) 
                if len(wf) > 100:
                    wf = random.sample(wf, 100)
                

                formatted_names = {
                    'W_W': wf, 'W_M': wm,
                }

            # only black and white 
            if id =='karvonen' or id == 'seshadri':
                wm = names[names['group'] == 'white_male']['name'].tolist()
                wf = names[names['group'] == 'white_female']['name'].tolist()
                bm = names[names['group'] == 'black_male']['name'].tolist()
                bf = names[names['group'] == 'black_female']['name'].tolist()
                
                
                if len(wm) > 100:
                    # sample 100 names from wm
                    wm = random.sample(wm, 100) 
                if len(wf) > 100:
                    wf = random.sample(wf, 100)
                if len(bm) > 100:
                    bm = random.sample(bm, 100)
                if len(bf) > 100:
                    bf = random.sample(bf, 100)
                formatted_names = {
                    'W_W': wf, 'B_W': bf, 'W_M': wm, 'B_M': bm,
                }



        return formatted_names

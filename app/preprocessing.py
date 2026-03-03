import pandas as pd

def generate_course_diff_column( df : pd.DataFrame ) -> pd.DataFrame :
    
    """-This function takes the entire dataframe as an argument\n
       -Generates the course_difficulty column\n
       -Returns The updated dataframe"""
    
    course_difficulty_map = {
    "Diploma": 0,
    "Bachelor of Arts": 1,
    "Bachelor of Commerce": 2,
    "Bachelor of Business Administration": 3,
    "Bachelor of Computer Applications": 4,
    "Bachelor of Science": 5,
    "Bachelor of Technology": 6
    }
    df['course_diff'] = df['course'].map( course_difficulty_map )
    
    return df

def encode_sleep_quality( df ) :
    df['sleep_quality'] = df['sleep_quality'].map({
        'poor' : 0, 'average' : 1, 'good' : 2
    })
    return df
    
def encode_facility_rating( df ) :
    df['facility_rating'] = df['facility_rating'].map({
        'low' : 0, 'medium' : 1, 'high' : 2
    })
    return df
    
def encode_exam_difficulty( df ) :

    df['exam_difficulty'] = df['exam_difficulty'].map({
        'easy' : 0, 'moderate' : 1, 'hard' : 2
    })
    return df


def apply_1_hot_encoding(df, full_df):
    # Wir machen die Spalten zu "Categorical", damit get_dummies alle Optionen sieht
    for col in ['gender', 'course', 'internet_access', 'study_method']:
        df[col] = pd.Categorical(df[col], categories=full_df[col].unique())
    
    df = pd.get_dummies(df, 
               columns=['gender', 'course', 'internet_access', 'study_method'],
               dtype=int)
    return df

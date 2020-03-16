import pandas as pd
import sqlalchemy
import sys


def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    return messages, categories

def clean_data(messages, categories):
    # split the single column into 36 different columns
    categories_split = categories['categories'].str.split(';',expand=True)

    # extract the different column names by splitting at the - and taking the first value
    category_colnames = []
    for value in categories_split.head(1).values[0]:
        column_name, number = value.split('-')
        category_colnames.append(column_name)

    # concatenate the original ids to the new split up dataframe and rename the columns accordingly
    categories_expanded = pd.concat([categories['id'], categories_split], axis=1)
    categories_expanded.columns = ['id', *category_colnames]

    # Apply function above to all values in dataframe and convert to numeric
    categories_expanded = categories_expanded.applymap(fix_categories_data)
    categories_expanded = categories_expanded.apply(lambda x: pd.to_numeric(x),axis=1)

    # Merge datasets
    df_merged = messages.merge(categories_expanded, how='inner', on='id')

    #Remove duplicates
    # We will arbitrarily pick the first value as there are only 68 cases out of 26248
    # where the duplicated rows have different values
    df_clean = df_merged.loc[~df_merged['id'].duplicated(keep='first')]
    return df_clean

#function to apply to all cells to strip the unnecessary words
def fix_categories_data(cell):
    if '-' in str(cell): #Condition put in to avoid errors with the id column
        return cell.split('-')[1]
    else:
        return cell

def save_data(df, database_filename):
    engine = sqlalchemy.create_engine('sqlite:///'+database_filename)
    df.to_sql('DisasterResponseTable', engine, index=False)

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        messages, categories = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(messages, categories)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
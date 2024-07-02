

import ast
def extract_text_from_dict_string(dict_string):
    prompt_string = dict_string["prompt"]
    # Convert the string representation of a dictionary to an actual dictionary
    dict_obj = ast.literal_eval(prompt_string)
    # Access the 'text' key from the dictionary
    return dict_obj['text']

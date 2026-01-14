import argparse
import json
import re

def substr_from_first_lonely_hash(s: str) -> str:
    m = re.search(r'(?<!#)#(?!#)', s)
    if not m:
        print("error #")
        return "No single '#' found whose neighbors are not '#'"
    return s[m.start():]

def extract_after_outline(input_string):
    
    input_string = input_string[5:]
    outline_index = input_string.find('<OUTLINE>')
    if outline_index != -1:
        outline_index = outline_index+len('<OUTLINE>')
        
    if outline_index == -1:
        outline_index = input_string.find('# OUTLINE')
        if outline_index !=-1:
            outline_index = outline_index+len('# OUTLINE')
            
    if outline_index == -1:
        new_string = substr_from_first_lonely_hash(input_string)
        return new_string
    
    after_outline = input_string[outline_index:]
    first_hash_index = after_outline.find('#')
    if first_hash_index == -1:
        return "No # found after <OUTLINE>"
    return after_outline[first_hash_index:]

def remove_after_last_to_be_filled(input_string):
    last_index = input_string.rfind('<TO BE FILLED>')
    if last_index == -1:
        return input_string 
    return input_string[:last_index + len('<TO BE FILLED>')]

def check_string_format(input_string):
    
    if not input_string.startswith('#'):
        print("The string does not start with '#'")
        raise SystemExit("The string does not start with '#'") 
    
    if not input_string.endswith('<TO BE FILLED>'):
        print("The string does not end with '<TO BE FILLED>'")
        raise SystemExit("The string does not end with '<TO BE FILLED>'")
    
    print("String format is correct!")

def process_json_file(json_file):
    data_list = []
    with open(json_file, 'r', encoding='utf-8') as file:
        for line in file:
            data_list.append(json.loads(line.strip()))
        
    for entry in data_list:
        result = extract_after_outline(entry['init_page'])
        
        if result == "No single '#' found whose neighbors are not '#'":
            entry['init_page'] = 'None'
        else:
            result = remove_after_last_to_be_filled(result)
            check_string_format(result)
            entry['init_page'] = result
    
    return data_list


def main():
    parser = argparse.ArgumentParser(description="Direct batch vLLM page generator")
    parser.add_argument(
        "--json_file",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--out_file",
        type=str,
        default=None,
    )
    args = parser.parse_args()
    print(args)
    
    data_list = process_json_file(args.json_file)
    with open(args.out_file, 'w', encoding='utf-8') as file:
        for item in data_list:
            json.dump(item, file, ensure_ascii=False)
            file.write('\n')
            
if __name__ == "__main__":
    main()
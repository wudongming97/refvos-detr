import json

json_name = 'meta_expressions_valid.json'
txt_name_all_frames = 'meta_expressions_valid.txt'
txt_name_first_frame = 'meta_expressions_valid_first_frame.txt'

file = open(txt_name_all_frames, 'w')
with open(json_name) as f:
    pop_data = json.load(f)
    pop_data = pop_data['videos']
    for pop_dict in pop_data:
        video_id = str(pop_dict)
        for object in pop_data[pop_dict]['objects']:
            object_id = object
            object_category = pop_data[pop_dict]['objects'][object]['category']
            expression_all = pop_data[pop_dict]['objects'][object]['expressions']
            for expression in expression_all:
                temp = video_id + ' ' + object_id + ' ' + str(object_category) + ' ' + str(expression)
                file.write(temp + '\n')
            # expression_first = pop_data[pop_dict]['objects'][object]['expressions_first_frame']
            # frames_id = pop_data[pop_dict]['objects'][object]['expressions_first_frame']
    file.close()

file = open(txt_name_first_frame, 'w')
with open(json_name) as f:
    pop_data = json.load(f)
    pop_data = pop_data['videos']
    for pop_dict in pop_data:
        video_id = str(pop_dict)
        for object in pop_data[pop_dict]['objects']:
            object_id = object
            object_category = pop_data[pop_dict]['objects'][object]['category']
            expression_all = pop_data[pop_dict]['objects'][object]['expressions_first_frame']
            for expression in expression_all:
                temp = video_id + ' ' + object_id + ' ' + str(object_category) + ' ' + str(expression)
                file.write(temp + '\n')
            # expression_first = pop_data[pop_dict]['objects'][object]['expressions_first_frame']
            # frames_id = pop_data[pop_dict]['objects'][object]['expressions_first_frame']
    file.close()
import json

json_name = '../data/Youtube-VOS/train/meta.json'
txt_name = '../data/Youtube-VOS/train/ImageSets/train.txt'

file = open(txt_name, 'w')
with open(json_name) as f:
    pop_data = json.load(f)
    pop_data = pop_data['videos']
    for pop_dict in pop_data:
        video_id = str(pop_dict)
        file.write(video_id + '\n')
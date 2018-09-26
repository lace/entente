def get_height(v):
    return v[1]

def sort_landmarks_by_height(landmark_dict):
    from collections import OrderedDict
    return OrderedDict(sorted(
      landmark_dict.items(),
      key=lambda item: get_height(item[1])))

def print_landmarks(landmark_dict, units='', precision=3):
    sorted_landmarks = sort_landmarks_by_height(landmark_dict)
    value_label = 'height ({})'.format(units) if units else 'height'
    print '{:15} {:>12}'.format('name', value_label)
    print '-' * 36
    for name, vert in ((name, sorted_landmarks[name]) for name in reversed(sorted_landmarks)):
        print '{:15} {:>12.{}f}'.format(name, vert[1], precision)

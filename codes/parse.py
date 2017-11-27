import xml.etree.ElementTree as ET
import definition

flatten = lambda l: [item for sublist in l for item in sublist]

def _parse_file(filename):
    """
    :param filename:
    :return: dictionary: {Instance ID: Instance Object}

    Instance Object:
    Member variables: context [text]
                      questions: dictionary: {Question ID: Question Object}

    Question Object:
    Member variables: question [text]
                      correct_answer [Answer class object]
                      incorrect_answer [Answer class object]

    Answer Object:
    Member variables: id
                      answer [text]
    """
    all_instances = {}
    root = ET.parse(filename).getroot()
    current_instance_id = None
    for elem in root.iter():
        if elem.tag == 'instance': current_instance_id = elem.attrib['id']
        if elem.tag == 'text':
            all_instances[current_instance_id] = definition.Instance(elem.text)
        if elem.tag == 'question':
            all_instances[current_instance_id]._add_questions(elem.attrib['id'], definition.Question(elem.attrib['text']))
        if elem.tag == 'answer':
            all_instances[current_instance_id]._add_answer(elem.attrib['correct']=='True', definition.Answer(elem.attrib['id'], elem.attrib['text']))
    return all_instances

def _get_word_to_index(vocab):
    word2index = {'<PAD>': 0, '<UNK>': 1}
    for vo in vocab:
        if vo not in word2index.keys():
            word2index[vo] = len(word2index)
    index2word = {v: k for k, v in word2index.items()}
    return word2index, index2word

def _get_vocab(data):
    vocab = []
    for key in data:
        vocab += flatten(data[key].context) + flatten(flatten(data[key]._get_all_questions()))
    return list(set(vocab))

def _get_training_data(data):
    zipped_data = []
    for key in data:
        zipped_data += data[key]._get_zipped_data()
    return zipped_data
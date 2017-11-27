from nltk.tokenize import word_tokenize,sent_tokenize
from copy import deepcopy

class Answer(object):
    def __init__(self, id, answer):
        self.id = id
        self.answer = word_tokenize(answer.lower())

    def _get_id(self): return self.id

    def _get_answer(self): return ' '.join(self.answer)

    def __str__(self): return self._get_id()+' : '+self._get_answer()

class Question(object):
    def __init__(self, ques):
        self.question = word_tokenize(ques.lower())
        self.correct_answer = None
        self.incorrect_answer = None

    def _add_answer(self, correct, answer):
        if correct: self.correct_answer = answer
        else: self.incorrect_answer = answer

    def __str__(self):
        return ' '.join(self.question) + '\n\t' + self.correct_answer.__str__() + "\n\t" + self.incorrect_answer.__str__()

class Instance(object):
    def __init__(self, context):
        self.context = [word_tokenize(sent) for sent in sent_tokenize(context.lower())]
        self.questions = {}
        self.current_question_id = None

    def _add_questions(self, id, question_obj):
        self.current_question_id = id
        self.questions[id] = question_obj

    def _add_answer(self, correct, answer):
        self.questions[self.current_question_id]._add_answer(correct, answer)

    def _get_all_questions(self):
        all_questions = []
        for id in self.questions:
            all_questions.append([self.questions[id].question, self.questions[id].correct_answer.answer, self.questions[id].incorrect_answer.answer])
        return all_questions

    def _get_zipped_data(self):
        data = []
        for id in self.questions:
            if (self.questions[id].correct_answer._get_id() == '0'):
                data.append([deepcopy(self.context), self.questions[id].question, self.questions[id].correct_answer.answer, self.questions[id].incorrect_answer.answer, 0])
            else:
                data.append([deepcopy(self.context), self.questions[id].question, self.questions[id].incorrect_answer.answer, self.questions[id].correct_answer.answer, 1])
        return data

    def __str__(self):
        ans = ' '.join(self.context[0]) + ' ..... \n'
        for id in self.questions:
            ans += "Question ID : " + id + '\n' + self.questions[id].__str__() + '\n'
        return ans.encode('ascii', 'ignore')



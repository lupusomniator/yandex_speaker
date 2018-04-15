import codecs
from  ufal.udpipe import Pipeline, Model, ProcessingError
from string import punctuation
default_model_path = "russian-syntagrus-ud-2_0-170801.udpipe"

'''
Предоставляет возможность токинизации, лемматизации и разметки по частям речи
Предобработка не убирает пунктуацию и стоп-слова
'''
class TextPreprocessor:
    '''
    Конструктор класса
    model_path - путь до модели языка для UdPipe
    '''
    def __init__(self,model_path: str = default_model_path):
        self.model_path = model_path
        print("Загрузка модели слов...")
        self.model = Model.load(model_path)
        if self.model == None:
            print("Не удалось загрузить модель %s", model_path)
        else:
            print("Загрузка модели слов прошла успешно!")
            self.pipeline = Pipeline(self.model, # Используемая языковая модель
                                     "horizontal", # Формат входной строки
                                     Pipeline.DEFAULT,# Используемый тэггер.
                                                      #По умолчанию Universal PoS (но это не точно)
                                     Pipeline.DEFAULT,# Используемый парсер
                                     "conllu")# Формат выходной строки
    '''
    Отделить все не буквенные символы пробелом от остальных
    '''
    def space_punctuation(self, s: str) -> str:
        return ''.join(c if c not in punctuation else ' %s '%(c) for c in s)
    '''
    Выполняет полную предобработку одного предложения
    Примечание: 
        Предложением считается любая строка. Все знаки пунктуации,
    включая знаки окончания преложения в тексте остаются и также размечаются.
    Результат:
        None, если операция не может быть завершена
        иначе - список нормализованных слов
    '''
    def preprocess_sentence(self,sentence: str) -> list:
            sentence = sentence.lower().replace('ё','е')
            sentence = self.space_punctuation(sentence)
            err = ProcessingError()
            processed = self.pipeline.process(sentence, err)
            if err.occurred():
                print(sentence)
                print("Произошла ошибка во время предобработки текста: ")
                print(err.message)
            else:
                result = []
                splited = processed.split('\n')
                for line in splited[2:]:
                    # Пропус пустых строк и специальных строк (начинаются с '#')
                    if line=='' or line[0]=='#':
                        continue
                    tokens = line.split('\t')
                    norm_form = tokens[2]
                    tag = tokens[3]
                    result.append(norm_form + '_' + tag)
            return result
    '''
    Преобразует список слов в текст
    '''
    def list_to_text(self, sentence_list: list) -> str:
        return ' '.join(word for word in sentence_list)
                    
            

#%%
if __name__ == '__main__':
    model_path = default_model_path
    tp = TextPreprocessor(model_path)
#%%             
if __name__ == '__main__':
    output = open ("Processed_sent3.txt","w")
    
    with codecs.open("OpenSubtitles2016.en-ru.ru", "r", "utf_8_sig") as f:
        for i, l in enumerate(f):
            new_cent = tp.list_to_text(tp.preprocess_sentence(l))
            try:
                output.write(new_cent + "\n")
            except:
                pass
            if (i%1000 == 0):
                print(i)
from kivy.app import App
from kivy.lang import Builder
from kivy.uix.screenmanager import Screen, ScreenManager
from tensorflow import keras
from tensorflow.keras.preprocessing import image_dataset_from_directory
import numpy as np
import os


class MainMenu(Screen):
    pass


class IntroducingScreen(Screen):
    pass


class CameraScreen(Screen):
    def capture(self):
        camera = self.ids['camera']
        camera.export_to_png("detect/photos/IMG_dish.png")


class ResultScreen(Screen):
    def detect(self):
        classes = ('Блины с творогом', 'Горячий бутерброд с ветчинной и сыром', 'Гречка с курицей', 'жёлтый комок',
                   'Зефир в стаканчиках', 'Котлета с овощами', 'Котлета с макаронами', 'Крабовый салат',
                   'Кулич', 'Маффин', 'Торт "Медовик"', 'Пицца',
                   'Ролл с курицей барбекю', 'Салат "Лидия"', 'Салат "Оливье"', 'Сырники с малиновой начинкой',
                   'Торт крем брюле в стаканчиках', 'Торт морковно-карамельный', 'Торт шоколадно-карамельный',
                   'Фило ролл с курицей',
                   'Хотдог', 'Шарлотта', 'Эклер с варёной сгущёнкой', 'Фило Ролл с ветчиной и сыром')
        kkal = ('196.9', '290.6', '164.6', '0',
                '326', '330', '213', '196.3',
                '313', '376', '400', '230',
                '526', '150', '172.1', '223',
                '281.3', '323.6', '320.1', '443.8',
                '542', '337.9', '385', '221')

        protein = ('10.43', '18.48', '11.3', '0',
                   '0.87', '9.78', '15.43', '6.74',
                   '7.61', '7.93', '5.43', '10.87',
                   '27.72', '7.61', '7.5', '15.33',
                   '6.74', '4.57', '8.04', '18.91',
                   '343.48', '5.54', '8.59', '11.63')

        fats = ('12.39', '25.97', '9.25', '0'
                                          '0.15', '41.79', '17.01', '13.43',
                '13.13', '21.04', '29.85', '14.93',
                '39.4', '17.9', '21.49', '12.84',
                '23.43', '33.28', '22.09', '25.22',
                '52.24', '28.06', '44.48', '15.07')

        carboh = ('14.82', '11.58', '12.16', '0',
                  '57.41', '12.23', '9.35', '16.19',
                  '40.94', '39.42', '35.97', '18.71',
                  '33.6', '3.61', '2.73', '16.04',
                  '20.43', '19.06', '28.56', '39.93',
                  '29.5', '27.05', '27.34', '15.68')

        batch_size = 256
        image_size = (100, 100)

        model = keras.models.load_model("food_model.h5")
        x_test = image_dataset_from_directory('detect',
                                              batch_size=batch_size,
                                              image_size=image_size)
        prediction = model.predict(x_test)
        abc = np.argmax(prediction, axis=1)
        self.dish.text = classes[abc[0] - 1]
        self.kkal.text = kkal[abc[0] - 1]
        self.protein.text = protein[abc[0] - 1]
        self.fats.text = fats[abc[0] - 1]
        self.carboh.text = carboh[abc[0] - 1]
        os.remove('detect/photos/IMG_dish.png')


class WindowManager(ScreenManager):
    pass


kv = Builder.load_file('my.kv')


class DetectCaloriesUrFU(App):
    def build(self):
        return kv


if __name__ == '__main__':
    DetectCaloriesUrFU().run()
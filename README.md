# Multimodal emotion classification

Мультимодальная классфикация эмоций по голосу и выражению лица человека с помощью глубокого обучения.

Модель для извлечения признаков из изображения лица обучалась на датасете [FER-2013](https://www.kaggle.com/datasets/msambare/fer2013) без класса disgusted.

Модель для классификации обучалась на датасете [RAVDESS](https://zenodo.org/record/1188976#.YnoqInVBx9D). Перед обучением классы calm и normal были объеденены.

>Livingstone SR, Russo FA (2018) The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS): A dynamic, multimodal set of facial and vocal expressions in North American English. PLoS ONE 13(5): e0196391. [https://doi.org/10.1371/journal.pone.0196391](https://doi.org/10.1371/journal.pone.0196391).


## Запуск приложения

Клонируем репозиторий:
```
git clone https://github.com/Kulallador/emotion.git
```

Создаем виртуальную среду
```
python -m venv env
```

Активируем среду
```
Для windows: 
env\Scripts\Activate

Для linux:
source env/bin/activate
```

Устанавливаем зависимости
```
pip install -r requirements.txt
```

Запускаем приложение 
```
python app.py
```





<h1 align="center">Решение для соревнования <a href="https://hacks-ai.ru/championships/758453">Всероссийский чемпионат, кейс РЖД</a>

## Разработка модели для сегментации объектов на планах помещений по изображениям 

###  Чтобы воспроизвести результаты, полученные в ходе решения данной задачи: Чтобы воспроизвести результаты, полученные в ходе решения данной задачи:

#### Разделение изображений на светлые и тёмные:
```
python split_dark_light.py --dataset-path <путь к папке "images" в датасете>
```

####  Обучение моделей
```
python main.py --cfg configs/segformer_1024_b4_g16_adamW_cosine.yml
python main.py --cfg configs/segformer_1024_b4_adamW_cosine.yml
python main.py --cfg configs/segformer_864_b4_8_cosine_light.yml
python main.py --cfg configs/segformer_864_b4_8_cosine_dark.yml
python main.py --cfg configs/segformer_512_b3_42_with_augs_batch16_fix_cosine_clean.yml
```

#### Генерация предсказаний

```
python generate_predictions.py --cfg experiments/segformer_864_b4_8_cosine_light/segformer_864_b4_8_cosine_light.yml  --checkpoint-path <Путь до чекпоинта>
python generate_predictions.py --cfg experiments/segformer_864_b4_8_cosine_light/segformer_864_b4_8_cosine_light.yml  --checkpoint-path <Путь до чекпоинта> --scale 0.741
python generate_predictions.py --cfg experiments/segformer_864_b4_8_cosine_light/segformer_864_b4_8_cosine_light.yml  --checkpoint-path <Путь до чекпоинта> --scale 0.651

python generate_predictions.py --cfg experiments/segformer_864_b4_8_cosine_dark/segformer_864_b4_8_cosine_dark.yml  --checkpoint-path <Путь до чекпоинта>
python generate_predictions.py --cfg experiments/segformer_864_b4_8_cosine_dark/segformer_864_b4_8_cosine_dark.yml  --checkpoint-path <Путь до чекпоинта> --scale 0.741
python generate_predictions.py --cfg experiments/segformer_864_b4_8_cosine_dark/segformer_864_b4_8_cosine_dark.yml  --checkpoint-path <Путь до чекпоинта> --scale 0.651

python generate_predictions.py --cfg experiments/segformer_512_b3_42_with_augs_batch16_fix_cosine_clean/segformer_512_b3_42_with_augs_batch16_fix_cosine_clean.yml  --checkpoint-path <Путь до чекпоинта>
python generate_predictions.py --cfg experiments/segformer_512_b3_42_with_augs_batch16_fix_cosine_clean/segformer_512_b3_42_with_augs_batch16_fix_cosine_clean.ymlsegformer_512_b3_42_with_augs_batch16_fix_cosine_clean.yml  --checkpoint-path <Путь до чекпоинта> --scale 0.741
python generate_predictions.py --cfg experiments/segformer_512_b3_42_with_augs_batch16_fix_cosine_clean/segformer_512_b3_42_with_augs_batch16_fix_cosine_clean.yml  --checkpoint-path <Путь до чекпоинта> --scale 0.651

python generate_predictions.py --cfg experiments/segformer_1024_b4_g16_adamW_cosine/segformer_1024_b4_g16_adamW_cosine.yml  --checkpoint-path <Путь до чекпоинта>
python generate_predictions.py --cfg experiments/segformer_1024_b4_g16_adamW_cosine/segformer_1024_b4_g16_adamW_cosine.yml  --checkpoint-path <Путь до чекпоинта> --scale 0.741
python generate_predictions.py --cfg experiments/segformer_1024_b4_g16_adamW_cosine/segformer_1024_b4_g16_adamW_cosine.yml  --checkpoint-path <Путь до чекпоинта> --scale 0.635
```

#### Объедините папки predictions/segformer_864_b4_8_cosine_dark и predictions/segformer_864_b4_8_cosine_light

#### Усреднение предсказаний и выдача результирующей маски
```
python create_submission_ensemble.py
```

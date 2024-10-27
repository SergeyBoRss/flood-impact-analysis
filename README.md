https://drive.google.com/file/d/1xhFOyOnpVPRRMlWYE8_M4XAw5l-FKPva/view?usp=sharing -веса обученой модели

python process_image.py path/to/input.tif path/to/output.tif --threshold 0.08 --model path/to/deeplabv3_trained_epoch_65_2.pth - пример запуска инференса

в inference.ipynb содержится пример вывод картинок и анализ метрик

# Аргументы
image_path (обязательный): Путь к входному 10-канальному TIF изображению.
output_path (обязательный): Путь для сохранения бинарной маски.
--threshold (необязательный): Порог для бинаризации маски (по умолчанию 0.08).
--model (обязательный): Путь до весов модели.

Позволяет обрабатывать 10-канальные TIF изображения и генерировать бинарные маски с использованием модели сегментации DeepLabV3

get_flooded_buildings.py - подсчет затопленых зданий
buildings_path = "dataset/train/osm/geo.geojson"
water_mask_path = "dataset/train/masks/mask.tif" - аргументы


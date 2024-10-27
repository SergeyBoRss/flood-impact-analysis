import geopandas as gpd
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import shape
from rasterio.features import shapes

buildings_path = "dataset/train/osm/geo.geojson"
water_mask_path = "dataset/train/masks/mask.tif"

# Загрузка GeoJSON файла с постройками
buildings = gpd.read_file(buildings_path)

# Загрузка TIFF файла с маской воды
with rasterio.open(water_mask_path) as src:
    water_mask = src.read(1)
    transform = src.transform  # Трансформация для привязки координат
    crs = src.crs  # Получение CRS изображения

# Преобразуем маску в бинарный формат (предполагаем, что вода отмечена значением 1)
water_mask = (water_mask == 1).astype(np.uint8)

# Конвертируем массив маски в полигоны
water_shapes = []
for geom, value in shapes(water_mask, mask=water_mask, transform=transform):
    if value == 1:  # Только области, помеченные как вода
        water_shapes.append(shape(geom))

# Создаем GeoDataFrame из форм воды
water_gdf = gpd.GeoDataFrame(geometry=water_shapes, crs=crs)

if buildings.crs != water_gdf.crs:
    buildings = buildings.to_crs(water_gdf.crs)

water_sindex = water_gdf.sindex

def intersects_water(building):
    possible_matches_index = list(water_sindex.intersection(building.bounds))
    possible_matches = water_gdf.iloc[possible_matches_index]
    return any(building.intersects(water) for water in possible_matches.geometry)

# Применяем функцию к каждому зданию
buildings['is_flooded'] = buildings.geometry.apply(intersects_water)


# Создаем отдельный GeoDataFrame для затопленных построек
flooded_buildings = buildings[buildings['is_flooded']]

print(len(flooded_buildings))

# Настройка отображения
fig, ax = plt.subplots(figsize=(10, 10))

# Отображаем маску воды
with rasterio.open(water_mask_path) as src:
    water_mask_img = src.read(1)
    water_mask_img = (water_mask_img == 1).astype(np.uint8)  # Убеждаемся, что вода - это 1
    extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
    ax.imshow(water_mask_img, cmap='Blues', extent=extent, alpha=0.5)

# Отображаем все постройки
buildings.plot(ax=ax, color='lightgrey', edgecolor='lightgrey', linewidth=0.5, label="Buildings")

# Отображаем затопленные постройки красным цветом
flooded_buildings.plot(ax=ax, color='red', edgecolor='red', linewidth=0.5, label="Flooded Buildings")

plt.title("Buildings and Flooded Buildings Overlayed on Water Mask")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()

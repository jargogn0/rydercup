from flask import Flask, render_template, request, jsonify
import os
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.merge import merge
import zipfile
from PIL import Image
import folium
from folium.plugins import MiniMap
import geopandas as gpd
import ipywidgets as widgets
from IPython.display import display, clear_output

app = Flask(__name__)


predicted_dir = "./predicted"
zipped_npy_path = os.path.join(predicted_dir, 'predicted_images.zip')

@app.route('/ndvi')
def ndvi_page():
    dir_path = "."

    return render_template('ndvi.html')
                    

#---MAPS--------------------------------------------------------------------------

# Directory containing the predicted images
predicted_dir = "./predicted"
predicted_image_files = sorted([os.path.join(predicted_dir, f) for f in os.listdir(predicted_dir) if f.endswith('.tif')])

# Constants
PIXEL_AREA = 30 * 30  # Assuming each pixel represents 30m x 30m
AVERAGE_CARBON_DENSITY_MATURE = 200  # Average carbon density for mature forests (tonnes per hectare)
AVERAGE_CARBON_DENSITY_YOUNG = 50  # Average carbon density for young/regrowing forests (tonnes per hectare)

forest_cover_area = []
total_carbon_stock = []
regrowth_areas = []
deforested_areas = []

previous_forest_cover = None
for f in predicted_image_files:
    with rasterio.open(f) as src:
        forest_cover = src.read(1) > 0.5
        
        # Calculate forest cover area
        forest_area = np.sum(forest_cover) * PIXEL_AREA / (10**6)  # in sq.km
        forest_cover_area.append(forest_area)
        
        # Calculate carbon stock
        carbon_stock = forest_area * AVERAGE_CARBON_DENSITY_MATURE  # in tonnes
        total_carbon_stock.append(carbon_stock)
        
        # If there's a previous year's data, calculate regrowth and deforested areas
        if previous_forest_cover is not None:
            regrowth = np.logical_and(~previous_forest_cover, forest_cover)
            deforestation = np.logical_and(previous_forest_cover, ~forest_cover)
            regrowth_areas.append(np.sum(regrowth) * PIXEL_AREA / (10**6))
            deforested_areas.append(np.sum(deforestation) * PIXEL_AREA / (10**6))
        else:
            regrowth_areas.append(0)
            deforested_areas.append(0)
        
        previous_forest_cover = forest_cover

# Carbon emissions due to deforestation
carbon_emissions = np.array(deforested_areas) * AVERAGE_CARBON_DENSITY_MATURE
# Visualization
years = list(range(2013, 2021))


# Constants and Map Initialization
center_coords = [13.0667, -12.7167]
MAPBOX_ACCESS_TOKEN = 'pk.eyJ1IjoiZGJjb29wZXIxOSIsImEiOiJjbGlveWZyeGgwNHNzM2xucWtmeHRtdjRjIn0.eR5g-CGcSLPyW_d_x-BAKw'
MAPBOX_SATELLITE_URL = f"https://api.mapbox.com/styles/v1/mapbox/satellite-streets-v11/tiles/256/{{z}}/{{x}}/{{y}}@2x?access_token={MAPBOX_ACCESS_TOKEN}"
MAPBOX_HYDROLOGY_URL = f"https://api.mapbox.com/styles/v1/mapbox/outdoors-v11/tiles/256/{{z}}/{{x}}/{{y}}@2x?access_token={MAPBOX_ACCESS_TOKEN}"
image_bounds = [[13.0667 - 0.6, -12.7167 - 0.6], [13.0667 + 0.4, -12.7167 + 0.4]]

# Add Niokolo-Koba Park shapefile
shp_path = "./niokolokoshp/WDPA_WDOECM_Sep2023_Public_2580_shp_0.zip"
niokolokoba_gdf = gpd.read_file(shp_path)
style_function_park = lambda x: {'fillColor': '#32CD32', 'color': '#32CD32', 'fillOpacity': 0.5, 'weight': 0.5}

# Define the paths for each year
layers_by_year = {}
for year in range(2013, 2021):
    overlay_image_path = f"./colored_images/forest_overlay_{year}.png"
    emission_image_path = f"./emission_colored_images/carbon_emission_{year}.png"
    stock_image_path = f"./stock_colored_images/carbon_stock_{year}.png"
    layers_by_year[year] = (overlay_image_path, emission_image_path, stock_image_path)


# Define alert thresholds
FOREST_COVER_DECREASE_THRESHOLD = 10  # sq.km
FOREST_COVER_INCREASE_THRESHOLD = 10  # sq.km
CARBON_EMISSION_THRESHOLD = 1000  # tonnes
CARBON_STOCK_DECREASE_THRESHOLD = 1000  # tonnes
CARBON_STOCK_INCREASE_THRESHOLD = 1000  # tonnes



@app.route('/detailed_map')
def detailed_map():
    
    # Constants
    PIXEL_AREA = 30 * 30
    AVERAGE_CARBON_DENSITY_MATURE = 200
    DEFORESTATION_THRESHOLD = 5  # 5% of the forest cover lost
    CARBON_STOCK_DECREASE_THRESHOLD = 10 # 10% decrease in carbon stock

    # Constants for Mapbox
    MAPBOX_ACCESS_TOKEN = 'pk.eyJ1IjoiZGJjb29wZXIxOSIsImEiOiJjbGlveWZyeGgwNHNzM2xucWtmeHRtdjRjIn0.eR5g-CGcSLPyW_d_x-BAKw'
    MAPBOX_SATELLITE_URL = f"https://api.mapbox.com/styles/v1/mapbox/satellite-streets-v11/tiles/256/{{z}}/{{x}}/{{y}}@2x?access_token={MAPBOX_ACCESS_TOKEN}"
    MAPBOX_HYDROLOGY_URL = f"https://api.mapbox.com/styles/v1/mapbox/outdoors-v11/tiles/256/{{z}}/{{x}}/{{y}}@2x?access_token={MAPBOX_ACCESS_TOKEN}"

    # Initialize folium map
    center_coords = [13.0667, -12.7167]
    m = folium.Map(location=center_coords, zoom_start=10, tiles=None)

    # Add MiniMap
    minimap = MiniMap()
    m.add_child(minimap)

    # Add base layers
    folium.TileLayer(MAPBOX_HYDROLOGY_URL, attr="Mapbox Hydrology", name="Hydrology").add_to(m)
    folium.TileLayer(MAPBOX_SATELLITE_URL, attr="Mapbox Satellite", name="Satellite Imagery").add_to(m)
    folium.TileLayer('openstreetmap').add_to(m)
    folium.TileLayer('Stamen Terrain').add_to(m)
    folium.TileLayer('Stamen Toner').add_to(m)

    # Directory for predicted images
    predicted_dir = "./predicted"
    predicted_image_files = sorted([os.path.join(predicted_dir, f) for f in os.listdir(predicted_dir) if f.endswith('.tif')])

    previous_forest_cover = None
    carbon_emission_layers = []
    carbon_stock_layers = []

    for f in predicted_image_files:
        with rasterio.open(f) as src:
            forest_cover = src.read(1) > 0.5

            if previous_forest_cover is not None:
                deforestation = np.logical_and(previous_forest_cover, ~forest_cover)
                carbon_emission = deforestation.astype(np.float32) * AVERAGE_CARBON_DENSITY_MATURE
                carbon_emission_layers.append(carbon_emission)
            else:
                carbon_emission_layers.append(np.zeros_like(forest_cover, dtype=np.float32))

            carbon_stock = forest_cover.astype(np.float32) * AVERAGE_CARBON_DENSITY_MATURE
            carbon_stock_layers.append(carbon_stock)

            previous_forest_cover = forest_cover

    # Image bounds
    image_bounds = [[13.0667 - 0.5, -12.7167 - 0.5], [13.0667 + 0.5, -12.7167 + 0.5]]

    for year in range(2013, 2021):
        overlay_image_path = f"./colored_images/forest_overlay_{year}.png"
        emission_image_path = f"./emission_colored_images/carbon_emission_{year}.png"
        stock_image_path = f"./stock_colored_images/carbon_stock_{year}.png"

        img = folium.raster_layers.ImageOverlay(name=f"Predicted Forest {year}", image=overlay_image_path, bounds=image_bounds, opacity=0.6, show=False)
        img.add_child(folium.Popup(f'Predicted Forest Cover for {year}'))
        img.add_to(m)

        emission_img = folium.raster_layers.ImageOverlay(name=f"Carbon Emission {year}", image=emission_image_path, bounds=image_bounds, opacity=0.6, show=False)
        emission_img.add_to(m)

        stock_img = folium.raster_layers.ImageOverlay(name=f"Carbon Stock {year}", image=stock_image_path, bounds=image_bounds, opacity=0.6, show=False)
        stock_img.add_to(m)

    # Load Niokolo-Koba Park shapefile using geopandas
    shp_path = "./niokolokoshp/WDPA_WDOECM_Sep2023_Public_2580_shp_0.zip"
    niokolokoba_gdf = gpd.read_file(shp_path)

    # Add Niokolo-Koba Park shapefile to folium map with a distinct color
    style_function_park = lambda x: {'fillColor': '#32CD32', 'color': '#32CD32', 'fillOpacity': 0.5, 'weight': 0.5}
    folium.GeoJson(niokolokoba_gdf, style_function=style_function_park, name="Niokolo-Koba National Park").add_to(m)

    import re

    alerts = []

    # Regular expression pattern to match a four-digit year
    year_pattern = re.compile(r'\b\d{4}\b')

    for f in predicted_image_files:
        # Extracting the year using regex
        match = year_pattern.search(f)
        if not match:
            continue  # Skip files that don't have a year in the filename
        
        year = int(match.group())

        with rasterio.open(f) as src:
            forest_cover = src.read(1) > 0.2

            if previous_forest_cover is not None:
                deforestation = np.logical_and(previous_forest_cover, ~forest_cover)
                deforestation_percentage = (deforestation.sum() / previous_forest_cover.sum()) * 100
                print(f"Deforestation percentage for {year}: {deforestation_percentage:.2f}%")
                
                if deforestation_percentage > DEFORESTATION_THRESHOLD:
                    alerts.append(f"ALERT for {year}: More than 5% of the forest was cleared.")

                carbon_stock = forest_cover.astype(np.float32) * AVERAGE_CARBON_DENSITY_MATURE
                if previous_year_carbon_stock is not None:
                    change_in_carbon_stock = (carbon_stock.sum() - previous_year_carbon_stock) / previous_year_carbon_stock * 100
                    print(f"Change in carbon stock for {year}: {change_in_carbon_stock:.2f}%")
                    
                    if change_in_carbon_stock < -CARBON_STOCK_DECREASE_THRESHOLD:
                        alerts.append(f"ALERT for {year}: Carbon stock decreased by more than 10% compared to the previous year.")

                previous_year_carbon_stock = carbon_stock.sum()

            previous_forest_cover = forest_cover


    # Add alerts to the folium map
    for alert in alerts:
        folium.Marker(
            location=center_coords,
            popup=alert,
            icon=folium.Icon(color="red", icon="exclamation-circle")
        ).add_to(m)
        
    # Enhanced legend with hover functionality
    legend_html = """
    <div style="position: fixed; bottom: 50px; left: 50px; z-index:9999; font-size:14px; background-color: white; padding: 10px; border-radius: 5px;">
        <b>Legend:</b><br>
        <i class="fa fa-square fa-1x" style="color:green"></i> Predicted Forest<br>
        <i class="fa fa-square fa-1x" style="color:red"></i> Carbon Emission (due to deforestation)<br>
        <i class="fa fa-square fa-1x" style="color:blue"></i> Carbon Stock<br>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    # Layer Control
    folium.LayerControl().add_to(m)

    m

    map_html = m._repr_html_()
    return render_template('map_display.html', map_html=map_html)
    


def generate_alerts(year, image_type):
    """Generate alerts based on forest data for the given year and image type."""
    alerts = []

    idx = years.index(year)
    
    if image_type == "Predicted Forest":
        # Alert for significant decrease in forest cover
        if idx > 0 and (forest_cover_area[idx-1] - forest_cover_area[idx]) >= FOREST_COVER_DECREASE_THRESHOLD:
            alerts.append("⚠️ Dramatic Deforestation Detected!")
        # Alert for reforestation
        if idx > 0 and (forest_cover_area[idx] - forest_cover_area[idx-1]) >= FOREST_COVER_INCREASE_THRESHOLD:
            alerts.append("🌱 Significant Reforestation Detected!")

    elif image_type == "Carbon Emission":
        # Alert for increase in carbon emission 
        if idx > 0 and carbon_emissions[idx] > carbon_emissions[idx-1]:
            alerts.append(f"⚠️ Increase in Carbon Emission")
            alerts.append(f"> {carbon_emissions[idx]:.2f} tonnes!")
        # Alert for decrease in carbon emission if there's reforestation or increase in carbon stock
        elif idx > 0 and (forest_cover_area[idx] - forest_cover_area[idx-1] >= FOREST_COVER_INCREASE_THRESHOLD or total_carbon_stock[idx] > total_carbon_stock[idx-1]):
            alerts.append("🌱 Decrease in Carbon Emission")
            alerts.append(f"< {carbon_emissions[idx]:.2f} tonnes!")

    elif image_type == "Carbon Stock":
        # Alert for decrease in carbon stock
        if idx > 0 and total_carbon_stock[idx] < total_carbon_stock[idx-1]:
            alerts.append(f"⚠️ Decrease in Carbon Stock")
            alerts.append(f"< {total_carbon_stock[idx]:.2f} tonnes!")
        # Alert for increase in carbon stock
        if idx > 0 and total_carbon_stock[idx] > total_carbon_stock[idx-1]:
            alerts.append(f"🌱 Increase in Carbon Stock")
            alerts.append(f"> {total_carbon_stock[idx]:.2f} tonnes!")

    return alerts

    
    # Add base layers
    folium.TileLayer(MAPBOX_HYDROLOGY_URL, attr="Mapbox Hydrology", name="Hydrology").add_to(m)
    folium.TileLayer(MAPBOX_SATELLITE_URL, attr="Mapbox Satellite", name="Satellite Imagery").add_to(m)
    folium.TileLayer('openstreetmap').add_to(m)
    folium.TileLayer('Stamen Terrain').add_to(m)
    folium.TileLayer('Stamen Toner').add_to(m)
    
def display_map(year_slider, image_type):
    # Create a new map
    m = folium.Map(location=center_coords, zoom_start=8, tiles=None)
    
    # Add base layers
    folium.TileLayer(MAPBOX_HYDROLOGY_URL, attr="Mapbox Hydrology", name="Hydrology").add_to(m)
    # Add base layers
    #folium.TileLayer(MAPBOX_HYDROLOGY_URL, attr="Mapbox Hydrology", name="Hydrology").add_to(m)
    #folium.TileLayer(MAPBOX_SATELLITE_URL, attr="Mapbox Satellite", name="Satellite Imagery").add_to(m)
    #folium.TileLayer('openstreetmap').add_to(m)
    #folium.TileLayer('Stamen Terrain').add_to(m)
    #folium.TileLayer('Stamen Toner').add_to(m)

    # Add park shapefile
    folium.GeoJson(niokolokoba_gdf, style_function=style_function_park, name="Niokolo-Koba National Park").add_to(m)
    
    # Retrieve the image paths for the selected year
    overlay_image_path, emission_image_path, stock_image_path = layers_by_year.get(year_slider, ('', '', ''))

    # Display the corresponding layer based on the image type selected
    if image_type == "Predicted Forest" and os.path.exists(overlay_image_path):
        folium.raster_layers.ImageOverlay(name=f"Predicted Forest {year_slider}", image=overlay_image_path, bounds=image_bounds, opacity=0.6).add_to(m)
    elif image_type == "Carbon Emission" and os.path.exists(emission_image_path):
        folium.raster_layers.ImageOverlay(name=f"Carbon Emission {year_slider}", image=emission_image_path, bounds=image_bounds, opacity=0.6).add_to(m)
    elif image_type == "Carbon Stock" and os.path.exists(stock_image_path):
        folium.raster_layers.ImageOverlay(name=f"Carbon Stock {year_slider}", image=stock_image_path, bounds=image_bounds, opacity=0.6).add_to(m)

    # Save the map
    map_filename = f'map_{image_type}.html'
    m.save(os.path.join('templates', map_filename))

    return map_filename


#-----------------------------------
dir_path = "."


import matplotlib.animation as animation
from IPython.display import HTML

@app.route('/animation')
def mamita():

    return render_template('animation.html')


#-----------------------------------Model
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory, jsonify, Response, stream_with_context
import os
import tensorflow as tf
import rasterio
import numpy as np
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from PIL import Image

app.secret_key = 'adjingone'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PREDICTED_FOLDER'] = 'predicted'

# Logging setup
logging.basicConfig(level=logging.INFO)

# Model and directories setup
model_path = "./forest_model.h5"
model = tf.keras.models.load_model(model_path)

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.mkdir(app.config['UPLOAD_FOLDER'])

predicted_dir = "./predicted"
if not os.path.exists(predicted_dir):
    os.mkdir(predicted_dir)

patch_size = 50
accuracies, precisions, recalls, f1_scores = [], [], [], []

from rasterio.enums import Resampling

def resize_image(src_path, width, height):
    with rasterio.open(src_path) as src:
        data = src.read(
            out_shape=(src.count, height, width),
            resampling=Resampling.bilinear
        )
        return data[0]

def reconstruct_image(patches, original_shape):
    reconstructed = np.zeros(original_shape)
    idx = 0
    for i in range(0, original_shape[0] - patch_size, patch_size):
        for j in range(0, original_shape[1] - patch_size, patch_size):
            reconstructed[i:i+patch_size, j:j+patch_size] = patches[idx]
            idx += 1
    return reconstructed

@app.route('/main', endpoint='main_page')
def main():
    return "Main Page"

@app.route('/model')
def modelpage():
    return render_template('model.html')

abort_prediction = False

@app.route('/abort_prediction', methods=['POST'])
def abort_prediction_request():
    global abort_prediction
    abort_prediction = True
    return jsonify(success=True)

@app.route('/predict', methods=['POST'])
def predict():
    def generate():
        global abort_prediction

        b3_images = request.files.getlist('b3_images[]')
        b4_images = request.files.getlist('b4_images[]')

        if not b3_images or not b4_images:
            flash('Please upload B3 satellite image(s) and B4 satellite image(s).')
            return redirect(url_for('modelpage'))

        predicted_image_paths = []
        total_images = len(b3_images)
        
        for index, (b3_image, b4_image) in enumerate(zip(b3_images, b4_images), start=1):
            if abort_prediction:
                abort_prediction = False
                yield f"data: {{\"progress\": {int((index/total_images) * 100)}, \"message\": \"Prediction aborted!\"}}\n\n"
                break

            yield f"data: {{\"progress\": {int((index/total_images) * 100)}, \"message\": \"Processing image {index}/{total_images}\"}}\n\n"
            
            b3_path = os.path.join(app.config['UPLOAD_FOLDER'], b3_image.filename)
            b4_path = os.path.join(app.config['UPLOAD_FOLDER'], b4_image.filename)
            b3_image.save(b3_path)
            b4_image.save(b4_path)

            try:
                with rasterio.open(b4_path) as nir_src:
                    nir_data = nir_src.read(1)
                with rasterio.open(b3_path) as red_src:
                    red_data = red_src.read(1)

                nir_data_resized = resize_image(b4_path, red_data.shape[1], red_data.shape[0])
                ndvi_data = (nir_data_resized - red_data) / (nir_data_resized + red_data + 1e-8)
                truths = (ndvi_data > 0.5).astype(np.int32).ravel()

                predicted_patches = []
                for i in range(0, ndvi_data.shape[0] - patch_size, patch_size):
                    for j in range(0, ndvi_data.shape[1] - patch_size, patch_size):
                        patch = ndvi_data[i:i+patch_size, j:j+patch_size]
                        reshaped_patch = patch.reshape((1, patch_size, patch_size, 1))

                        if patch.shape[0] == 50 and patch.shape[1] == 50:
                            logging.info(f"Running prediction on patch at position ({i}, {j})")
                            prediction = model.predict(reshaped_patch)
                            predicted_patches.append(prediction[0][0])
                            logging.info(f"Prediction completed for patch at position ({i}, {j})")

                predicted_image = reconstruct_image(predicted_patches, ndvi_data.shape)
                predicted_image_path = os.path.join(predicted_dir, f'predicted_{b3_image.filename.split("_")[0]}.png')
                Image.fromarray((predicted_image * 255).astype(np.uint8)).save(predicted_image_path)

                predictions = (predicted_image > 0.5).astype(np.int32).ravel()
                accuracies.append(accuracy_score(truths, predictions))
                precisions.append(precision_score(truths, predictions))
                recalls.append(recall_score(truths, predictions))
                f1_scores.append(f1_score(truths, predictions))

                predicted_image_paths.append(predicted_image_path)
            except Exception as e:
                yield f"data: {{\"progress\": {int((index/total_images) * 100)}, \"message\": \"Error processing image {index}: {str(e)}\"}}\n\n"

        accuracy = np.mean(accuracies)
        precision = np.mean(precisions)
        recall = np.mean(recalls)
        f1 = np.mean(f1_scores)

        yield f"data: {{\"progress\": 100, \"message\": \"Prediction complete!\", \"predicted_image_urls\": {predicted_image_paths}, \"accuracy\": {accuracy}, \"precision\": {precision}, \"recall\": {recall}, \"f1\": {f1}}}\n\n"

    return Response(stream_with_context(generate()), content_type='text/event-stream')

@app.route('/predicted/<filename>')
def predicted_image(filename):
    return send_from_directory(app.config['PREDICTED_FOLDER'], filename)

#-------------

def on_change(year_slider, image_type):
    display_map(year_slider, image_type)

# Create interactive widgets
image_dropdown = widgets.Dropdown(options=["Predicted Forest", "Carbon Emission", "Carbon Stock"], value="Predicted Forest", description='Image Type:')
widgets.interactive(on_change, year_slider=widgets.IntSlider(min=2013, max=2020, step=1, value=2013, description='Year:'), image_type=image_dropdown)

    

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/map_display')
def map_display():
    year = request.args.get('year', 2013, type=int)
    image_type = request.args.get('image_type', "Predicted Forest")
    map_filename = display_map(year, image_type)
    alerts = generate_alerts(year, image_type)
    with open(os.path.join('templates', map_filename), 'r') as f:
        mapHTML = f.read()
    return jsonify({"mapHTML": mapHTML, "alerts": alerts})

app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0


if __name__ == '__main__':
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    app.run(debug=True,port=5001)

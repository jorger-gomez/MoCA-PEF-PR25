# Clock Drawing Segmentation with Deep Learning in the MoCA Test
#
# This module implements a Flask web application for analyzing clock drawings from the
# Montreal Cognitive Assessment (MoCA) test using deep learning techniques. The system
# enables automated analysis to assist in cognitive impairment detection through a user-friendly
# web interface where users can draw or upload clock images.
#
# Authors and Contact Information:
# - Diego Aldahir Tovar Ledesma - diego.tovar@udem.edu
# - Jorge Rodrigo Gómez Mayo - jorger.gomez@udem.edu
#
# Organization: Universidad de Monterrey
# First created: February 2025

"""
Flask web application for the NeuroLink clock analysis system.

This application provides a web interface for analyzing clock drawings 
from the Montreal Cognitive Assessment (MoCA) test to detect early signs
of cognitive impairment. It allows users to draw or upload clock images,
which are then processed by a neural network model to perform segmentation
and analysis.

The application includes routes for the main pages (home, evaluation, about),
as well as endpoints for processing images and displaying results.
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for, abort
import os
import base64
import uuid
from datetime import datetime
import numpy as np
import cv2
from models.clock_analyzer import analyze_clock_image
import traceback
import shutil

# Initialize Flask application
app = Flask(__name__)

# Configuration constants
UPLOAD_FOLDER = os.path.join('WebPage_v2/static', 'uploads')
RESULTS_FOLDER = os.path.join('WebPage_v2/static', 'results')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Dictionary to temporarily store analysis results (would use a database in production)
analysis_results = {}


def allowed_file(filename):
    """Check if the uploaded file has an allowed extension.
    
    Args:
        filename (str): The name of the uploaded file.
        
    Returns:
        bool: True if the file extension is allowed, False otherwise.
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def home():
    """Render the home page.
    
    Returns:
        rendered_template: The rendered home page template.
    """
    return render_template('inicio.html')


@app.route('/evaluacion')
def evaluacion():
    """Render the evaluation page where users can draw or upload clock images.
    
    Returns:
        rendered_template: The rendered evaluation page template.
    """
    return render_template('evaluacion.html')


@app.route('/nosotros')
def nosotros():
    """Render the about page with information about the project.
    
    Returns:
        rendered_template: The rendered about page template.
    """
    return render_template('nosotros.html')


@app.route('/procesar', methods=['POST'])
def procesar_imagen():
    """Process an image received as base64 data.
    
    This endpoint receives a base64-encoded image from a canvas drawing,
    saves it to the uploads folder, processes it with the clock analysis model,
    and returns an ID to access the results.
    
    Returns:
        JSON response: Contains the analysis_id if successful, or an error message.
    """
    if 'image_data' not in request.form:
        return jsonify({'error': 'No image received'}), 400
    
    try:
        # Get image data
        image_data = request.form['image_data']
        source = request.form.get('source', 'unknown')
        
        # Remove base64 header to save the image
        if 'base64,' in image_data:
            image_data = image_data.split('base64,')[1]
        
        # Generate unique name for the image
        image_id = str(uuid.uuid4())
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_filename = f"{source}_{timestamp}_{image_id}.png"
        
        # Use absolute path to ensure correct location
        app_dir = os.path.dirname(os.path.abspath(__file__))
        upload_folder = os.path.join(app_dir, UPLOAD_FOLDER)
        results_folder = os.path.join(app_dir, RESULTS_FOLDER)
        
        # Ensure folders exist
        os.makedirs(upload_folder, exist_ok=True)
        os.makedirs(results_folder, exist_ok=True)
        
        # Save the original image
        image_path = os.path.join(UPLOAD_FOLDER, image_filename)
        with open(image_path, "wb") as f:
            f.write(base64.b64decode(image_data))
        
        # Verify the image was saved correctly
        if not os.path.exists(image_path) or os.path.getsize(image_path) == 0:
            return jsonify({'error': 'Error saving the image'}), 500
            
        # Print debug information
        print(f"Image saved at: {image_path}")
        print(f"File size: {os.path.getsize(image_path)} bytes")
        
        # Process the image with the analysis model
        analysis_result, _ = analyze_clock_image(image_path)
        
        # Save the result to access from the results page
        analysis_id = str(uuid.uuid4())
        
        # Add original filename to the results dictionary
        analysis_result['image_filename'] = image_filename
        analysis_result['timestamp'] = timestamp
        
        # If there's an original_filename in analysis_result, use it as display_image
        if 'original_filename' in analysis_result:
            analysis_result['display_image'] = analysis_result['original_filename']
        else:
            analysis_result['display_image'] = image_filename
        
        # Save in the results dictionary
        analysis_results[analysis_id] = analysis_result
        
        return jsonify({'analysis_id': analysis_id})
        
    except Exception as e:
        traceback.print_exc()
        print(f"Error processing the image: {str(e)}")
        return jsonify({'error': 'Error processing the image'}), 500


@app.route('/procesar_archivo', methods=['POST'])
def procesar_archivo():
    """Process an uploaded image file.
    
    This endpoint receives an image file upload, saves it to the uploads folder,
    processes it with the clock analysis model, and returns an ID to access the results.
    
    Returns:
        JSON response: Contains the analysis_id if successful, or an error message.
    """
    if 'image_file' not in request.files:
        return jsonify({'error': 'No image received'}), 400
    
    try:
        # Get the image file
        image_file = request.files['image_file']
        
        # Generate unique name for the image
        image_id = str(uuid.uuid4())
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_filename = f"uploaded_{timestamp}_{image_id}.png"
        
        # Use absolute path to ensure correct location
        app_dir = os.path.dirname(os.path.abspath(__file__))
        upload_folder = os.path.join(app_dir, UPLOAD_FOLDER)
        results_folder = os.path.join(app_dir, RESULTS_FOLDER)
        
        # Ensure folders exist
        os.makedirs(upload_folder, exist_ok=True)
        os.makedirs(results_folder, exist_ok=True)
        
        # Save the original image
        image_path = os.path.join(UPLOAD_FOLDER, image_filename)
        image_file.save(image_path)
        
        # Verify the image was saved correctly
        if not os.path.exists(image_path) or os.path.getsize(image_path) == 0:
            return jsonify({'error': 'Error saving the image'}), 500
            
        print(f"Image saved at: {image_path}")
        print(f"File size: {os.path.getsize(image_path)} bytes")
        
        # Process the image with the clock analysis model
        analysis_result, _ = analyze_clock_image(image_path)
        
        # Save the result to access from the results page
        analysis_id = str(uuid.uuid4())
        
        # Add original filename to the results dictionary
        analysis_result['image_filename'] = image_filename
        analysis_result['timestamp'] = timestamp
        
        # If there's an original_filename in analysis_result, use it as display_image
        if 'original_filename' in analysis_result:
            analysis_result['display_image'] = analysis_result['original_filename']
        else:
            analysis_result['display_image'] = image_filename
        
        # Save in the results dictionary
        analysis_results[analysis_id] = analysis_result
        
        return jsonify({'analysis_id': analysis_id})
        
    except Exception as e:
        traceback.print_exc()
        print(f"Error processing the image: {str(e)}")
        return jsonify({'error': 'Error processing the image'}), 500


@app.route('/resultados')
def resultados():
    """Display the analysis results page.
    
    This route retrieves analysis results using the ID from the query parameter
    and renders the results template with the data.
    
    Returns:
        rendered_template: The rendered results page template with analysis data.
    """
    analysis_id = request.args.get('id')
    if not analysis_id or analysis_id not in analysis_results:
        abort(404)
    
    resultado = analysis_results[analysis_id]
    
    # Verify and ensure there's at least one image to display
    if 'analysis_filename' in resultado and 'display_image' not in resultado:
        resultado['display_image'] = resultado['analysis_filename']
    
    return render_template('resultados.html', resultado=resultado)


@app.errorhandler(404)
def page_not_found(e):
    """Handle 404 errors by rendering a custom 404 page.
    
    Args:
        e: The error that triggered the handler.
        
    Returns:
        rendered_template: The rendered 404 page template.
    """
    return render_template('404.html'), 404


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5050)

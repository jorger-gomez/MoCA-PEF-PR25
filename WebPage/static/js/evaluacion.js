/**
 * Clock Drawing Test Interface - NeuroLink Project
 * 
 * This JavaScript file implements the interactive drawing and image upload
 * functionality for the clock drawing test evaluation page. It provides
 * two input methods: drawing directly on a canvas or uploading an image file.
 * The file handles canvas setup, drawing events (mouse and touch), image uploads,
 * and submission of data to the server for analysis.
 * 
 * Authors:
 * - Diego Aldahir Tovar Ledesma - diego.tovar@udem.edu
 * - Jorge Rodrigo Gómez Mayo - jorger.gomez@udem.edu
 * 
 * Organization: Universidad de Monterrey
 * First created: April 2025
 */

document.addEventListener('DOMContentLoaded', function() {
    // DOM elements
    const drawCanvas = document.getElementById('drawingCanvas');
    const ctx = drawCanvas.getContext('2d');
    const drawInstruction = document.getElementById('draw-instruction');
    const clearButton = document.getElementById('clear-draw');
    const fileInput = document.getElementById('file');
    const imageCanvas = document.getElementById('imageCanvas');
    const imgCtx = imageCanvas.getContext('2d');
    const removeImageButton = document.getElementById('remove-image');
    const uploadLabel = document.querySelector('.upload-label');
    const submitButton = document.getElementById('submit-button');
    
    // State variables
    let drawingEnabled = false;  // Whether the drawing mode is active
    let imageUploaded = false;   // Whether an image has been uploaded
    let isDrawing = false;       // Whether user is currently drawing
    let lastX = 0;               // Last X position for drawing
    let lastY = 0;               // Last Y position for drawing

    // Initial canvas settings
    ctx.lineWidth = 3;           // Pen thickness
    ctx.lineCap = 'round';       // Round line ends
    ctx.lineJoin = 'round';      // Round line joins
    ctx.strokeStyle = '#000';    // Black color for drawing
    
    /**
     * Resizes canvas to match its container dimensions
     * 
     * @param {HTMLCanvasElement} canvas - The canvas to resize
     * @param {HTMLElement} container - The container element
     */
    function resizeCanvas(canvas, container) {
        if (container) {
            canvas.width = container.clientWidth;
            canvas.height = container.clientHeight;
        }
    }
    
    /**
     * Sets a white background for the canvas
     * 
     * @param {HTMLCanvasElement} canvas - The canvas element
     * @param {CanvasRenderingContext2D} context - The canvas 2D context
     */
    function setWhiteBackground(canvas, context) {
        context.fillStyle = '#FFFFFF';
        context.fillRect(0, 0, canvas.width, canvas.height);
    }

    /**
     * Updates the "Analyze" button state based on whether there's content to analyze
     */
    function updateSubmitButton() {
        if (drawingEnabled || imageUploaded) {
            submitButton.classList.remove('disabled');
        } else {
            submitButton.classList.add('disabled');
        }
    }

    /**
     * Resets the image upload interface state
     */
    function resetImageUpload() {
        imageCanvas.classList.add('hidden');
        removeImageButton.classList.add('hidden');
        fileInput.value = "";  // Reset file input
        uploadLabel.classList.remove('hidden');
        imageUploaded = false;
        updateSubmitButton();
    }

    // Event listener to initialize drawing when clicking on the canvas area
    drawCanvas.parentElement.addEventListener('click', function(e) {
        if (!drawingEnabled && !imageUploaded) {
            drawingEnabled = true;
            drawInstruction.style.opacity = "0";  // Hide instructions
            clearButton.classList.remove('hidden');  // Show reset button
            // Set white background when starting to draw
            setWhiteBackground(drawCanvas, ctx);
            updateSubmitButton();
        }
    });

    // Event listener for the clear/reset button
    clearButton.addEventListener('click', (e) => {
        e.stopPropagation();  // Prevent click from propagating to canvas
        // Clear and reset white background
        setWhiteBackground(drawCanvas, ctx);
        drawingEnabled = true;  // Allow drawing again after clearing
        updateSubmitButton();
    });

    // Mouse event listeners for drawing
    drawCanvas.addEventListener('mousedown', startDrawing);
    drawCanvas.addEventListener('mousemove', draw);
    drawCanvas.addEventListener('mouseup', stopDrawing);
    drawCanvas.addEventListener('mouseout', stopDrawing);
    
    // Touch event listeners for mobile devices
    drawCanvas.addEventListener('touchstart', handleTouchStart);
    drawCanvas.addEventListener('touchmove', handleTouchMove);
    drawCanvas.addEventListener('touchend', stopDrawing);

    /**
     * Starts the drawing process when mouse/touch is pressed
     * 
     * @param {Event} e - The mouse or touch event
     */
    function startDrawing(e) {
        if (!drawingEnabled) return;
        
        isDrawing = true;
        [lastX, lastY] = getCoordinates(e);
        
        // Draw a dot at the starting point
        ctx.beginPath();
        ctx.moveTo(lastX, lastY);
        ctx.lineTo(lastX, lastY);
        ctx.stroke();
    }

    /**
     * Continues drawing as mouse/touch moves
     * 
     * @param {Event} e - The mouse or touch event
     */
    function draw(e) {
        if (!isDrawing || !drawingEnabled) return;
        
        // Prevent scrolling on touch devices
        e.preventDefault();
        
        const [currentX, currentY] = getCoordinates(e);
        
        ctx.beginPath();
        ctx.moveTo(lastX, lastY);
        ctx.lineTo(currentX, currentY);
        ctx.stroke();
        
        [lastX, lastY] = [currentX, currentY];
    }

    /**
     * Stops the drawing process when mouse/touch is released
     */
    function stopDrawing() {
        isDrawing = false;
    }

    /**
     * Gets coordinates from mouse or touch event
     * 
     * @param {Event} e - The mouse or touch event
     * @returns {Array} Array containing [x, y] coordinates
     */
    function getCoordinates(e) {
        const rect = drawCanvas.getBoundingClientRect();
        
        if (e.type.includes('touch')) {
            return [
                e.touches[0].clientX - rect.left,
                e.touches[0].clientY - rect.top
            ];
        } else {
            return [
                e.clientX - rect.left,
                e.clientY - rect.top
            ];
        }
    }

    /**
     * Handles touch start events for mobile drawing
     * 
     * @param {TouchEvent} e - The touch event
     */
    function handleTouchStart(e) {
        e.preventDefault();  // Prevent zoom/scroll
        startDrawing(e);
    }

    /**
     * Handles touch move events for mobile drawing
     * 
     * @param {TouchEvent} e - The touch event
     */
    function handleTouchMove(e) {
        e.preventDefault();  // Prevent zoom/scroll
        draw(e);
    }

    // Image upload handling
    fileInput.addEventListener('change', (e) => {
        if (drawingEnabled) return;  // Don't allow upload if in drawing mode
        
        const file = e.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = (event) => {
                const img = new Image();
                img.src = event.target.result;
                img.onload = () => {
                    imageUploaded = true;
                    imageCanvas.classList.remove('hidden');
                    removeImageButton.classList.remove('hidden');
                    uploadLabel.classList.add('hidden');
                    // Clear and set white background
                    setWhiteBackground(imageCanvas, imgCtx);
                    
                    // Calculate dimensions to maintain aspect ratio
                    let width = img.width;
                    let height = img.height;
                    const maxWidth = imageCanvas.width;
                    const maxHeight = imageCanvas.height;
                    
                    if (width > height) {
                        if (width > maxWidth) {
                            height = height * (maxWidth / width);
                            width = maxWidth;
                        }
                    } else {
                        if (height > maxHeight) {
                            width = width * (maxHeight / height);
                            height = maxHeight;
                        }
                    }
                    
                    // Center the image on the canvas
                    const x = (maxWidth - width) / 2;
                    const y = (maxHeight - height) / 2;
                    
                    imgCtx.drawImage(img, x, y, width, height);
                    updateSubmitButton();
                };
            };
            reader.readAsDataURL(file);
        }
    });

    // Event listener for removing uploaded image
    removeImageButton.addEventListener('click', (e) => {
        e.stopPropagation();  // Prevent click propagation
        resetImageUpload();
    });

    /**
     * Sends the image as a file to the server
     * Uses the modern Blob API for improved performance and reliability
     */
    async function sendImageAsFile() {
        try {
            const formData = new FormData();
            
            if (drawingEnabled) {
                // Convert canvas to Blob
                drawCanvas.toBlob(blob => {
                    formData.append('image_file', blob, 'drawing.png');
                    
                    // Show loading indicator
                    submitButton.textContent = "Analizando...";
                    submitButton.classList.add('disabled');
                    
                    // Send to server
                    fetch('/procesar_archivo', {
                        method: 'POST',
                        body: formData
                    })
                    .then(response => response.json())
                    .then(result => {
                        if (result.error) {
                            throw new Error(result.error);
                        }
                        window.location.href = `/resultados?id=${result.analysis_id}`;
                    })
                    .catch(error => {
                        console.error("Error:", error);
                        alert(`Error: ${error.message || 'Ocurrió un error al procesar la imagen'}`);
                        submitButton.textContent = "Analizar Reloj";
                        submitButton.classList.remove('disabled');
                    });
                }, 'image/png', 1.0);
            } else if (imageUploaded) {
                // Use the file directly if available
                if (fileInput.files && fileInput.files[0]) {
                    formData.append('image_file', fileInput.files[0]);
                } else {
                    // Or convert the canvas to Blob
                    imageCanvas.toBlob(blob => {
                        formData.append('image_file', blob, 'uploaded.png');
                        
                        // Show loading indicator
                        submitButton.textContent = "Analizando...";
                        submitButton.classList.add('disabled');
                        
                        // Send to server
                        fetch('/procesar_archivo', {
                            method: 'POST',
                            body: formData
                        })
                        .then(response => response.json())
                        .then(result => {
                            if (result.error) {
                                throw new Error(result.error);
                            }
                            window.location.href = `/resultados?id=${result.analysis_id}`;
                        })
                        .catch(error => {
                            console.error("Error:", error);
                            alert(`Error: ${error.message || 'Ocurrió un error al procesar la imagen'}`);
                            submitButton.textContent = "Analizar Reloj";
                            submitButton.classList.remove('disabled');
                        });
                    }, 'image/png', 1.0);
                    return; // Exit because processing is asynchronous
                }
                
                // Show loading indicator
                submitButton.textContent = "Analizando...";
                submitButton.classList.add('disabled');
                
                // Send to server
                try {
                    const response = await fetch('/procesar_archivo', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const result = await response.json();
                    if (result.error) {
                        throw new Error(result.error);
                    }
                    window.location.href = `/resultados?id=${result.analysis_id}`;
                } catch (error) {
                    console.error("Error:", error);
                    alert(`Error: ${error.message || 'Ocurrió un error al procesar la imagen'}`);
                    submitButton.textContent = "Analizar Reloj";
                    submitButton.classList.remove('disabled');
                }
            }
        } catch (error) {
            console.error("Error:", error);
            alert(`Error: ${error.message || 'Ocurrió un error al procesar la imagen'}`);
            submitButton.textContent = "Analizar Reloj";
            submitButton.classList.remove('disabled');
        }
    }

    /**
     * Original method to send image data as base64 (fallback)
     * Used as a fallback if the Blob method fails
     */
    async function sendImageData() {
        try {
            const formData = new FormData();
            
            if (drawingEnabled) {
                // Get canvas data with maximum quality
                const canvasData = drawCanvas.toDataURL('image/png', 1.0);
                formData.append('image_data', canvasData);
                formData.append('source', 'drawing');
            } else if (imageUploaded) {
                // Get image canvas data with maximum quality
                const canvasData = imageCanvas.toDataURL('image/png', 1.0);
                formData.append('image_data', canvasData);
                formData.append('source', 'upload');
            }
            
            // Show loading indicator
            submitButton.textContent = "Analizando...";
            submitButton.classList.add('disabled');
            
            const response = await fetch('/procesar', {
                method: 'POST',
                body: formData
            });
            
            if (response.ok) {
                const result = await response.json();
                if (result.error) {
                    throw new Error(result.error);
                }
                window.location.href = `/resultados?id=${result.analysis_id}`;
            } else {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Ocurrió un error al procesar la imagen');
            }
        } catch (error) {
            console.error("Error:", error);
            alert(`Error: ${error.message || 'Ocurrió un error al procesar la imagen'}`);
            submitButton.textContent = "Analizar Reloj";
            submitButton.classList.remove('disabled');
        }
    }

    // Event listener for the submit button
    submitButton.addEventListener('click', async () => {
        if (submitButton.classList.contains('disabled')) return;
        
        // Try the file method first, and if it fails use the data method as fallback
        try {
            await sendImageAsFile();
        } catch (error) {
            console.error("Error with file method, trying with data method:", error);
            await sendImageData();
        }
    });

    /**
     * Adjusts canvas sizes and layouts on window resize
     */
    function adjustLayout() {
        // Adjust canvas to container size
        const canvasContainer = drawCanvas.parentElement;
        if (canvasContainer) {
            resizeCanvas(drawCanvas, canvasContainer);
            setWhiteBackground(drawCanvas, ctx);
        }
        
        const imgCanvasContainer = imageCanvas.parentElement;
        if (imgCanvasContainer) {
            resizeCanvas(imageCanvas, imgCanvasContainer);
            setWhiteBackground(imageCanvas, imgCtx);
        }
    }

    // Initialization
    adjustLayout();
    window.addEventListener('resize', adjustLayout);
    updateSubmitButton();
});
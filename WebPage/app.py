from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('inicio.html')  # Renderiza la página de inicio

@app.route('/registro')
def services():
    return render_template('registro.html')  # Renderiza la página de servicios

@app.route('/historial')
def about():
    return render_template('historial.html')  # Renderiza la página de About Us

@app.route('/nosotros')
def contact():
    return render_template('nosotros.html')  # Renderiza la página de Contact

if __name__ == '__main__':
    app.run(debug=True)

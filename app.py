from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from googletrans import Translator 
import sqlite3

app = Flask(__name__)
app.secret_key = "123"

# SQLite database setup
con = sqlite3.connect("database.db")
con.execute("CREATE TABLE IF NOT EXISTS customer (pid INTEGER PRIMARY KEY, name TEXT, email TEXT, password TEXT, number TEXT)")
con.close()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/training')
def training():
    return render_template('training.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        try:
            name = request.form['name']
            email = request.form['email']
            password = request.form['password']
            number = request.form['number']  # Retrieve the number from the form
            
            con = sqlite3.connect("database.db")
            cur = con.cursor()
            cur.execute("INSERT INTO customer (name, email, password, number) VALUES (?, ?, ?, ?)", (name, email, password, number))  # Insert number into the database
            con.commit()
            flash("Record Added Successfully", "success")
        except Exception as e:
            flash("Error Insert Operation: " + str(e), "danger")
        finally:
            con.close()
            return redirect(url_for("login"))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        name = request.form['name']
        password = request.form['password']
        con = sqlite3.connect("database.db")
        con.row_factory = sqlite3.Row
        cur = con.cursor()
        cur.execute("SELECT * FROM customer WHERE name=? AND password=?", (name, password))
        data = cur.fetchone()

        if data:
            session["username"] = data["name"]
            flash("Login Successful", "success")
            return redirect(url_for('translation'))
        else:
            flash("Username and Password Mismatch", "danger")
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash("Logged out successfully", "info")
    return redirect(url_for('login'))

@app.route('/translation', methods=['GET', 'POST'])
def translation():
    if 'username' in session:
        return render_template('translation.html')
    else:
        flash("You need to login first", "warning")
        return redirect(url_for('login'))

@app.route('/translate', methods=['POST'])
def translate():
    if request.method == 'POST':
        source_text = request.json['sourceText']
        translator = Translator()
        translated_text = translator.translate(source_text, src='en', dest='hi').text
        print("Trasnlated Text =",translated_text)
        return jsonify({'translated_text': translated_text})

if __name__ == '__main__':
    app.run(debug=True)

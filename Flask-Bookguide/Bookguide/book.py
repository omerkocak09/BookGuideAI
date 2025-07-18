from flask import Flask, render_template, flash, redirect, url_for, session, logging, request,jsonify
from flask_pymongo import PyMongo
from wtforms import Form, StringField, TextAreaField, PasswordField, validators
from passlib.hash import sha256_crypt
from functools import wraps
from bson.objectid import ObjectId
from pymongo.errors import ConnectionFailure
from datetime import datetime
import pickle
import pandas as pd
import numpy as np
import google.generativeai as genai

app = Flask(__name__)
app.secret_key = "bookguide"

# MongoDB bağlantı ayarları
app.config["MONGO_URI"] = "mongodb+srv://kocakomerfaruk40:lazerjet1453@cluster0.o2l0u.mongodb.net/bookguide"
mongo = PyMongo(app)

# 1. Modelleri Yükle
with open("Bookguide\\model\\kmeans_model.pkl", 'rb') as file:
    kmeans = pickle.load(file)

with open("Bookguide\\model\\tfidf_vectorizer.pkl", 'rb') as file:
    tfidf = pickle.load(file)

with open('Bookguide\\model\\label_encoder.pkl', 'rb') as file:
    label_encoder = pickle.load(file)

print(f'{label_encoder} yüklendi, {tfidf} yüklendi, {kmeans} yüklendi')
# Veri setini yükle ve işle
print("Veri seti yükleniyor...")
df = pd.read_csv("merged_books_with_images.csv")
print(f"Yüklenen veri seti boyutu: {df.shape}")
df['description'] = df['description'].fillna("Açıklama bulunmamaktadır.")
df['most_popular_genre'] = df['most_popular_genre'].fillna("fiction")

# Label encoding işlemi
df['genre_encoded'] = label_encoder.transform(df['most_popular_genre'])

# TF-IDF ile kitap açıklamalarını vektörleştir
description_vectors = tfidf.transform(df['description'])

# TF-IDF vektörleri ve 'genre_encoded' özelliğini birleştirme
combined_features = np.hstack((
    description_vectors.toarray(),
    df['genre_encoded'].values.reshape(-1, 1)
))

# K-Means modelini kullanarak kümeleri tahmin et
df['cluster'] = kmeans.predict(combined_features)

# 2. Öneri Fonksiyonu ML MODELİNİ KULLANARAK BENZER KİTAPLARI BULAN ASIL FONKSİYON
def recommend_books(title, df, kmeans_model, tfidf_model, label_encoder, top_n=5):
    """Benzer kitapları bulan ML fonksiyonu"""
    try:
        print(f"Aranan kitap: {title}")  # Debug için
        
        if title not in df['title'].values:
            print(f"'{title}' kitabı veri setinde bulunamadı")  # Debug için
            return None

        # Seçilen kitabın indeksini bul
        book_idx = df.index[df['title'] == title].tolist()[0]
        
        # Kitabın özelliklerini al
        book_description = df.loc[book_idx, 'description']
        book_genre = df.loc[book_idx, 'most_popular_genre']
        
        print(f"Kitap türü: {book_genre}")  # Debug için
        
        # Vektörleştirme ve tahmin
        book_vector = tfidf_model.transform([book_description])
        book_genre_encoded = label_encoder.transform([book_genre])
        
        book_features = np.hstack((
            book_vector.toarray(),
            book_genre_encoded.reshape(-1, 1)
        ))
        
        cluster_label = kmeans_model.predict(book_features)[0]
        print(f"Küme etiketi: {cluster_label}")  # Debug için
        
        # Benzer kitapları bul
        similar_books = df[df['cluster'] == cluster_label]['title'].tolist()
        similar_books.remove(title)
        
        print(f"Bulunan benzer kitap sayısı: {len(similar_books)}")  # Debug için
        
        return similar_books[:top_n] if len(similar_books) > top_n else similar_books
        
    except Exception as e:
        print(f"Öneri sisteminde hata: {str(e)}")  # Debug için
        return None

# Gemini API yapılandırması
GEMINI_API_KEY = "AIzaSyAe-JGamKxGdKgVfbXIdv0X5P05_EXGtL4"
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-pro')

try:
    # Bağlantıyı test et
    mongo.db.command('ping')
    print("MongoDB'ye başarıyla bağlanıldı!")
except ConnectionFailure:
    print("MongoDB bağlantısı başarısız!")

# Login decorator'ı aynı kalabilir
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "logged_in" in session:
            return f(*args, **kwargs)
        else:
            flash("Please log in to view this page.", "danger")
            return redirect(url_for("login"))
    return decorated_function

# Form sınıfları aynı kalabilir
class RegisterForm(Form):
    name = StringField("Name-Surname", validators=[validators.Length(min=4, max=25)])
    username = StringField("User name", validators=[validators.Length(min=5, max=35)])
    email = StringField("Email Address", validators=[validators.Email(message="Please Enter a Valid Email Address...")])
    password = PasswordField("Password:", validators=[
        validators.DataRequired(message="Please choose a password"),
        validators.EqualTo(fieldname="confirm", message="Your Password Does Not Match...")
    ])
    confirm = PasswordField("Verify Password")

class LoginForm(Form):
    username = StringField("User name")
    password = PasswordField("Password")

class BookForm(Form):
    title = StringField("Book Name", validators=[validators.Length(min=2, max=100)])
    author = StringField("Author", validators=[validators.Length(min=2, max=100)])
    description = StringField("Genre", validators=[validators.Length(min=2, max=100)])
    rating = StringField("Rate", validators=[validators.Length(min=1, max=5)])

class ChangePasswordForm(Form):
    old_password = PasswordField("Current Password", validators=[validators.DataRequired()])
    new_password = PasswordField("New Password", validators=[
        validators.DataRequired(),
        validators.EqualTo('confirm', message='Passwords do not match')
    ])
    confirm = PasswordField("New Password (Again)")

class ForgotPasswordForm(Form):
    email = StringField("Email Adresi", validators=[validators.Email(message="Please Enter a Valid Email Address...")])
    new_password = PasswordField("New Password", validators=[
        validators.DataRequired(),
        validators.EqualTo('confirm', message='Passwords do not match')
    ])
    confirm = PasswordField("New Password (Again)")

@app.route("/register", methods=["GET", "POST"])
def register():
    form = RegisterForm(request.form)
    if request.method == "POST" and form.validate():
        existing_user = mongo.db.users.find_one({"username": form.username.data})
        if existing_user is None:
            hashed_password = sha256_crypt.encrypt(form.password.data)
            mongo.db.users.insert_one({
                "name": form.name.data,
                "email": form.email.data,
                "username": form.username.data,
                "password": hashed_password
            })
            flash("You have successfully registered...", "success")
            return redirect(url_for("login"))
        flash("This username is already in use.", "danger")
    return render_template("register.html", form=form)

@app.route("/login", methods=["GET", "POST"])
def login():
    form = LoginForm(request.form)
    if request.method == "POST":
        user = mongo.db.users.find_one({"username": form.username.data})
        if user and sha256_crypt.verify(form.password.data, user["password"]):
            session["logged_in"] = True
            session["username"] = user["username"]
            session["name"] = user["name"]
            flash("You have successfully logged in...", "success")
            return redirect(url_for("index"))
        flash("The username or password is incorrect.", "danger")
    return render_template("login.html", form=form)

@app.route("/dashboard")
@login_required
def dashboard():
    # Kullanıcının kitaplarını çekerken book_id'yi ObjectId olarak saklayalım
    user_books = list(mongo.db.my_books.find({"username": session["username"]}))
    for book in user_books:
        if isinstance(book.get('book_id'), str):
            book['book_id'] = ObjectId(book['book_id'])
    return render_template("dashboard.html", books=user_books)

@app.route("/books/<int:page>")
@app.route("/books")
def show_books(page=0):
    category = request.args.get('category', None)
    search = request.args.get('search', None)
    skip = page * 25
    
    # Temel sorgu
    query = {}
    
    # Kategori filtresi düzeltmesi
    if category:
        # Virgülle ayrılmış kategorileri liste haline getir
        categories = [cat.strip() for cat in category.split(',')]
        # OR operatörü kullanarak kategorileri filtrele
        query["most_popular_genre"] = {
            "$regex": f"({'|'.join(categories)})", 
            "$options": "i"
        }
    
    # Arama filtresi
    if search:
        if query:
            query = {
                "$and": [
                    {"title": {"$regex": search, "$options": "i"}},
                    query
                ]
            }
        else:
            query["title"] = {"$regex": search, "$options": "i"}
    
    try:
        total_books = mongo.db.books.count_documents(query)
        books = list(mongo.db.books.find(query).sort("_id", -1).skip(skip).limit(25))
        total_pages = (total_books // 25)
        
        print(f"Query: {query}")  # Debug için
        print(f"Bulunan kitap sayısı: {len(books)}")  # Debug için
        
        return render_template("books.html", 
                             books=books, 
                             current_page=page,
                             total_pages=total_pages,
                             category=category)
                             
    except Exception as e:
        print(f"Hata oluştu: {str(e)}")  # Debug için
        flash("Kitapları getirirken bir hata oluştu.", "danger")
        return render_template("books.html", books=[], current_page=0, total_pages=0)

@app.route("/add_book", methods=["GET", "POST"])
@login_required
def add_book():
    form = BookForm(request.form)
    if request.method == "POST" and form.validate():
        mongo.db.books.insert_one({
            "title": form.title.data,
            "author": form.author.data,
            "description": form.description.data,
            "rating": form.rating.data,
            "username": session["username"]
        })
        flash("Book added successfully", "success")
        return redirect(url_for("books"))
    return render_template("add_book.html", form=form)

@app.route("/")
def index():
    return render_template("index.html")

# 3. API Endpoint buda frontend den gelen kitabı recommend_books() fonksiyonu ile benzer kitapları bulan fonksiyon
@app.route('/recommend', methods=['POST'])
def recommend():
    """API endpoint for book recommendations"""
    try:
        print("Recommend endpoint'i çağrıldı")
        data = request.json
        title = data.get('title', None)
        print(f"Aranan kitap: {title}")

        if not title:
            return jsonify({"error": "Kitap adı sağlanmadı"}), 400

        recommendations = recommend_books(title, df, kmeans, tfidf, label_encoder)
        print(f"Öneriler: {recommendations}")

        if recommendations is None:
            return jsonify({"error": f"'{title}' kitabı veri setinde bulunamadı"}), 404

        recommended_books = []
        for book_title in recommendations:
            book_details = mongo.db.books.find_one({"title": book_title})
            if book_details:
                recommended_books.append({
                    "_id": str(book_details["_id"]),
                    "title": book_details["title"],
                    "name": book_details.get("name", "Unknown Author"),
                    "average_rating_y": book_details.get("average_rating_y", 0),
                    "image_url": book_details.get("image_url", "/static/img/default-book.jpg")
                })

        print(f"MongoDB'den alınan kitap detayları: {len(recommended_books)}")  # Debug için
        
        return jsonify({
            "title": title,
            "recommendations": recommended_books
        })
        
    except Exception as e:
        print(f"Recommend endpoint hatası: {str(e)}")  # Debug için
        return jsonify({"error": "Bir hata oluştu"}), 500

@app.route("/book/<book_id>")
def book_detail(book_id):
    try:
        # ObjectId kontrolü ekleyelim
        if not ObjectId.is_valid(book_id):
            flash("Geçersiz kitap ID'si.", "danger")
            return redirect(url_for("show_books"))
            
        book = mongo.db.books.find_one({"_id": ObjectId(book_id)})
        if book:
            # Kitaba ait kullanıcı yorumlarını çek
            user_reviews = mongo.db.my_books.find({
                "book_id": ObjectId(book_id),
                "my_review": {"$exists": True, "$ne": ""}
            })
            return render_template("book_detail.html", book=book, user_reviews=list(user_reviews))
        else:
            flash("Book not found.", "danger")
            return redirect(url_for("show_books"))
    except Exception as e:
        print(f"An error occurred: {str(e)}")  # Hata ayıklama için log
        flash("An error occurred while displaying the book.", "danger")
        return redirect(url_for("show_books"))

@app.route("/logout")
def logout():
    session.clear()
    flash("You have successfully logged out.", "success")
    return redirect(url_for("index"))

@app.route("/add_to_mybooks/<book_id>", methods=["POST"])
@login_required
def add_to_mybooks(book_id):
    book = mongo.db.books.find_one({"_id": ObjectId(book_id)})
    
    if book:
        existing_book = mongo.db.my_books.find_one({
            "book_id": ObjectId(book_id),
            "username": session["username"]
        })
        
        if existing_book:
            flash("This book is already on your list!", "warning")
        else:
            mongo.db.my_books.insert_one({
                "book_id": ObjectId(book_id),
                "username": session["username"],
                "title": book["title"],
                "name": book["name"],
                "most_popular_genre": book["most_popular_genre"],
                "average_rating_y": book["average_rating_y"],
                "description": book["description"],
                "added_date": datetime.now(),
                "status": "unread",
                "my_review": ""
            })
            flash("The book has been successfully added to your list!", "success")
    else:
        flash("Book not found!", "danger")
    
    return redirect(url_for("book_detail", book_id=book_id))

@app.route("/remove_from_mybooks/<book_id>", methods=["POST"])
@login_required
def remove_from_mybooks(book_id):
    result = mongo.db.my_books.delete_one({
        "book_id": ObjectId(book_id),
        "username": session["username"]
    })
    
    if result.deleted_count > 0:
        flash("The book has been removed from your list.", "success")
    else:
        flash("The book was not found or the deletion failed.", "danger")
    
    return redirect(url_for("dashboard"))

@app.route("/update_book_status/<book_id>", methods=["POST"])
@login_required
def update_book_status(book_id):
    status = request.form.get('status')
    if status in ['read', 'unread']:
        mongo.db.my_books.update_one(
            {
                "book_id": ObjectId(book_id),
                "username": session["username"]
            },
            {"$set": {"status": status}}
        )
        flash("Book status updated.", "success")
    return redirect(url_for("dashboard"))

@app.route("/add_review/<book_id>", methods=["POST"])
@login_required
def add_review(book_id):
    review = request.form.get('review')
    mongo.db.my_books.update_one(
        {
            "book_id": ObjectId(book_id),
            "username": session["username"]
        },
        {"$set": {"my_review": review}}
    )
    flash("Your comment has been saved.", "success")
    return redirect(url_for("dashboard"))

@app.route("/change_password", methods=["GET", "POST"])
@login_required
def change_password():
    form = ChangePasswordForm(request.form)
    if request.method == "POST" and form.validate():
        user = mongo.db.users.find_one({"username": session["username"]})
        
        if user and sha256_crypt.verify(form.old_password.data, user["password"]):
            hashed_password = sha256_crypt.encrypt(form.new_password.data)
            mongo.db.users.update_one(
                {"username": session["username"]},
                {"$set": {"password": hashed_password}}
            )
            flash("Your password has been updated successfully.", "success")
            return redirect(url_for("dashboard"))
        else:
            flash("Your current password is incorrect.", "danger")
            
    return render_template("change_password.html", form=form)

@app.route("/forgot_password", methods=["GET", "POST"])
def forgot_password():
    form = ForgotPasswordForm(request.form)
    if request.method == "POST" and form.validate():
        user = mongo.db.users.find_one({"email": form.email.data})
        
        if user:
            hashed_password = sha256_crypt.encrypt(form.new_password.data)
            mongo.db.users.update_one(
                {"email": form.email.data},
                {"$set": {"password": hashed_password}}
            )
            flash("Your password has been updated successfully. You can log in with your new password.", "success")
            return redirect(url_for("login"))
        else:
            flash("No registered user was found with this email address.", "danger")
            
    return render_template("forgot_password.html", form=form)

@app.route("/rate_book/<book_id>", methods=["POST"])
@login_required
def rate_book(book_id):
    rating = request.form.get('rating')
    if rating and rating.isdigit() and 1 <= int(rating) <= 5:
        mongo.db.my_books.update_one(
            {
                "book_id": ObjectId(book_id),
                "username": session["username"]
            },
            {"$set": {"my_rating": int(rating)}}
        )
        flash("Your score has been saved.", "success")
    else:
        flash("Invalid score. Please enter a value between 1-5.", "danger")
    return redirect(url_for("dashboard"))

@app.route('/chat', methods=['POST'])
def chat():
    """Chatbot API endpoint with Gemini AI"""
    try:
        data = request.json
        user_message = data.get('message', '')
        
        # Gemini'ye gönderilecek prompt'u hazırla
        prompt = f"""Sen bir kitap asistanısın. Kitaplar, yazarlar ve edebiyat hakkında sorulara cevap veriyorsun.
        Soru: {user_message}
        Lütfen kitaplar hakkında bilgilendirici ve yardımcı bir yanıt ver."""
        
        # Gemini'den yanıt al
        response = model.generate_content(prompt)
        
        return jsonify({
            "response": response.text
        })
            
    except Exception as e:
        print(f"Chat hatası: {str(e)}")
        return jsonify({"error": "Bir hata oluştu"}), 500

if __name__ == "__main__":
    app.run(debug=True)
